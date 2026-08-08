"""
Unified Network Simulator GUI (Specification section 5.2).

Every panel is tied to a numbered requirement so the dashboard doubles as evidence that the
specification is implemented: the sliding windows and the two channel lanes visualise section
4.3, the waveform pair visualises section 3.1/3.2, and each metric cell is captioned with the
section it demonstrates.

Layout notes: the live tab is split into two columns so the whole dashboard fits on a short
screen without scrolling, because a real-time view is useless if the reader has to scroll away
from the animation to read the numbers. Every tab is still wrapped in a ScrollableFrame as a
fallback for very small displays or heavy DPI scaling.

The simulation always runs on a worker thread and reports back through a queue, so the window
keeps repainting while frames are in flight.
"""

import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from .security import SecurityLayer
    from .arq_protocol import SelectiveRepeatSimulation
    from .data_link_layer import Frame
    from .physical_layer import QAM16_DECISION_LEVELS, PhysicalLayerWireless
    from .performance_analysis import PerformanceAnalyzer
except ImportError:  # pragma: no cover
    from security import SecurityLayer
    from arq_protocol import SelectiveRepeatSimulation
    from data_link_layer import Frame
    from physical_layer import QAM16_DECISION_LEVELS, PhysicalLayerWireless
    try:
        from performance_analysis import PerformanceAnalyzer
    except ImportError:
        PerformanceAnalyzer = None


STATE_COLORS = {
    "UNSENT": "#E0E0E0",
    "SENT": "#FFD54F",
    "ACKED": "#4CAF50",
    "NAK": "#FF9800",
    "TIMED_OUT": "#EF5350",
    "ABANDONED": "#8E24AA",
}

TRANSIT_COLORS = {"DATA": "#1E88E5", "ACK": "#43A047", "NAK": "#FB8C00"}

CANVAS_BG = "#FAFAFA"
SCOPE_BG = "#101418"

PROTO_CANVAS_HEIGHT = 180
WAVE_CANVAS_HEIGHT = 64
FRAMING_CANVAS_HEIGHT = 40
CONST_CANVAS_SIZE = 172

FLAG_COLOR = "#FFD54F"
STUFF_COLOR = "#E040FB"
BODY_COLOR = "#00E676"

TRACE_I_TX = "#00E676"
TRACE_I_RX = "#FFA726"
TRACE_Q_TX = "#4FC3F7"
TRACE_Q_RX = "#CE93D8"


class ScrollableFrame(ttk.Frame):
    """
    Container whose contents may outgrow the window.

    Two behaviours matter here. The scrollbar only appears when the content genuinely does
    not fit, so a roomy window shows no chrome. And when there *is* spare room the inner
    frame is stretched to the viewport, otherwise panels asking to expand would be pinned to
    their requested height and leave dead space at the bottom of a large window.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self.canvas)
        self._item = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self._applied = (0, 0)

        self.interior.bind("<Configure>", lambda _event: self._refresh())
        self.canvas.bind("<Configure>", lambda _event: self._refresh())

    def _refresh(self):
        viewport_w = self.canvas.winfo_width()
        viewport_h = self.canvas.winfo_height()
        content_h = max(self.interior.winfo_reqheight(), viewport_h)

        if (viewport_w, content_h) != self._applied:
            self._applied = (viewport_w, content_h)
            self.canvas.itemconfigure(self._item, width=viewport_w, height=content_h)
            self.canvas.configure(scrollregion=(0, 0, viewport_w, content_h))

        overflowing = content_h > viewport_h
        if overflowing and not self.scrollbar.winfo_ismapped():
            self.scrollbar.pack(side="right", fill="y")
        elif not overflowing and self.scrollbar.winfo_ismapped():
            self.scrollbar.pack_forget()
            self.canvas.yview_moveto(0.0)

    @property
    def can_scroll(self) -> bool:
        return bool(self.scrollbar.winfo_ismapped())

    def scroll_by(self, delta: int):
        if self.can_scroll:
            notches = int(delta / 120) or (1 if delta > 0 else -1)
            self.canvas.yview_scroll(-notches * 3, "units")


class UnifiedSimulatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("End-to-End Secure Network Simulator - Unified Dashboard")
        self._apply_geometry()

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Caption.TLabel", font=("Segoe UI", 8), foreground="#5A5A5A")
        self.style.configure("Value.TLabel", font=("Segoe UI", 10, "bold"))

        self.live_queue: queue.Queue = queue.Queue()
        self.live_sim = None
        self.live_thread = None
        self._last_state = None

        self.term_queue: queue.Queue = queue.Queue()
        self.term_thread = None
        self._constellation_shown = False

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.scrollers = {}
        self.tab_live = self._add_tab("🎬 Live Transmission")
        self.tab_terminal = self._add_tab("💻 Terminal Simulation")
        self.tab_performance = self._add_tab("📈 Performance Analysis")

        self.setup_live_tab(self.tab_live.interior)
        self.setup_terminal_tab(self.tab_terminal.interior)
        self.setup_performance_tab(self.tab_performance.interior)

        self.root.bind_all("<MouseWheel>", self._on_global_wheel)

    def _apply_geometry(self):
        """Sizes the window against the real screen; hardcoded sizes break on scaled displays."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = max(min(1320, screen_w - 60), 820)
        height = max(min(940, screen_h - 110), 520)
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(820, 480)

    def _add_tab(self, label: str) -> ScrollableFrame:
        tab = ScrollableFrame(self.notebook)
        self.notebook.add(tab, text=label)
        self.scrollers[str(tab)] = tab
        return tab

    def _on_global_wheel(self, event):
        # Log panes scroll themselves; anywhere else the wheel should move the page.
        if isinstance(event.widget, tk.Text):
            return
        scroller = self.scrollers.get(self.notebook.select())
        if scroller is not None:
            scroller.scroll_by(event.delta)

    # ============================================================== live tab UI

    def setup_live_tab(self, parent):
        control = ttk.LabelFrame(parent, text=" ⚙️ Configuration ")
        control.pack(fill="x", padx=10, pady=(6, 4))
        for col in (1, 3, 5):
            control.columnconfigure(col, weight=1)

        ttk.Label(control, text="Message (§4.1 input):").grid(row=0, column=0, padx=4, pady=3, sticky="w")
        self.msg_entry = ttk.Entry(control)
        self.msg_entry.insert(0, "A secure modular live visual simulation of networks!")
        self.msg_entry.grid(row=0, column=1, columnspan=4, padx=4, pady=3, sticky="we")

        ttk.Button(control, text="📂 Load File", command=self.load_message_file, width=11) \
            .grid(row=0, column=5, padx=4, pady=3, sticky="w")

        ttk.Label(control, text="Key (§5.3):").grid(row=0, column=6, padx=4, pady=3, sticky="e")
        self.key_entry = ttk.Entry(control, width=15)
        self.key_entry.insert(0, "SecureXORKey")
        self.key_entry.grid(row=0, column=7, padx=4, pady=3, sticky="w")

        ber_box = ttk.Frame(control)
        ber_box.grid(row=0, column=8, padx=4, pady=3, sticky="w")
        ttk.Label(ber_box, text="Wired BER (§2):").pack(side="left", padx=(0, 3))
        self.ber_spin = ttk.Spinbox(ber_box, from_=0.0, to=0.2, increment=0.005, width=6,
                                    format="%.3f")
        self.ber_spin.set(0.005)
        self.ber_spin.pack(side="left")

        ttk.Label(control, text="Channel (§2):").grid(row=1, column=0, padx=4, pady=3, sticky="w")
        self.channel_var = tk.StringVar(value="wireless")
        self.channel_combo = ttk.Combobox(control, textvariable=self.channel_var,
                                          values=["wired", "wireless"], width=10, state="readonly")
        self.channel_combo.grid(row=1, column=1, padx=4, pady=3, sticky="w")
        self.channel_combo.bind("<<ComboboxSelected>>", self.update_channel_modes)

        ttk.Label(control, text="Mode (§3.1/§3.2):").grid(row=1, column=2, padx=4, pady=3, sticky="e")
        self.mode_var = tk.StringVar(value="BPSK")
        self.mode_combo = ttk.Combobox(control, textvariable=self.mode_var,
                                       values=["BPSK", "16-QAM"], width=10, state="readonly")
        self.mode_combo.grid(row=1, column=3, padx=4, pady=3, sticky="w")

        ttk.Label(control, text="Eb/N0 (dB):").grid(row=1, column=4, padx=4, pady=3, sticky="e")
        snr_box = ttk.Frame(control)
        snr_box.grid(row=1, column=5, columnspan=2, padx=4, pady=3, sticky="we")
        snr_box.columnconfigure(0, weight=1)
        self.snr_var = tk.DoubleVar(value=8.0)
        self.snr_scale = ttk.Scale(snr_box, from_=0.0, to=20.0, variable=self.snr_var,
                                   orient="horizontal", command=self._on_snr_change)
        self.snr_scale.grid(row=0, column=0, sticky="we")
        self.snr_label = ttk.Label(snr_box, text="8.0 dB", width=10)
        self.snr_label.grid(row=0, column=1, padx=(6, 0))

        ttk.Label(control, text="Payload/frame (B):").grid(row=1, column=7, padx=4, pady=3, sticky="e")
        self.chunk_spin = ttk.Spinbox(control, from_=2, to=32, width=4, state="readonly")
        self.chunk_spin.set(8)
        self.chunk_spin.grid(row=1, column=8, padx=4, pady=3, sticky="w")

        ttk.Label(control, text="Window W (§4.3):").grid(row=2, column=0, padx=4, pady=3, sticky="w")
        self.w_spin = ttk.Spinbox(control, from_=1, to=8, width=4, state="readonly")
        self.w_spin.set(4)
        self.w_spin.grid(row=2, column=1, padx=4, pady=3, sticky="w")

        ttk.Label(control, text="Timeout (ticks):").grid(row=2, column=2, padx=4, pady=3, sticky="e")
        self.timeout_spin = ttk.Spinbox(control, from_=2, to=20, width=4, state="readonly")
        self.timeout_spin.set(6)
        self.timeout_spin.grid(row=2, column=3, padx=4, pady=3, sticky="w")

        ttk.Label(control, text="Propagation (ticks):").grid(row=2, column=4, padx=4, pady=3, sticky="e")
        prop_box = ttk.Frame(control)
        prop_box.grid(row=2, column=5, padx=4, pady=3, sticky="w")
        self.prop_spin = ttk.Spinbox(prop_box, from_=1, to=6, width=4, state="readonly")
        self.prop_spin.set(2)
        self.prop_spin.pack(side="left")

        ttk.Label(prop_box, text="Max retries (§4.3):").pack(side="left", padx=(12, 3))
        self.retry_spin = ttk.Spinbox(prop_box, from_=1, to=12, width=4, state="readonly")
        self.retry_spin.set(5)
        self.retry_spin.pack(side="left")

        ttk.Label(control, text="Animation:").grid(row=2, column=6, padx=4, pady=3, sticky="e")
        self.speed_combo = ttk.Combobox(control, values=["Slow", "Normal", "Fast", "Instant"],
                                        width=8, state="readonly")
        self.speed_combo.set("Normal")
        self.speed_combo.grid(row=2, column=7, padx=4, pady=3, sticky="w")

        buttons = ttk.Frame(control)
        buttons.grid(row=2, column=8, padx=4, pady=3, sticky="e")
        self.start_btn = ttk.Button(buttons, text="🚀 Start", command=self.start_live_simulation, width=8)
        self.start_btn.pack(side="left", padx=1)
        self.stop_btn = ttk.Button(buttons, text="⏹ Stop", command=self.stop_live_simulation,
                                   width=7, state="disabled")
        self.stop_btn.pack(side="left", padx=1)

        # Two columns: the wide visualisations on the left, the readouts on the right.
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        body.columnconfigure(0, weight=3, minsize=560)
        body.columnconfigure(1, weight=2, minsize=330)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_protocol_panel(left)
        self._build_framing_panel(left)
        self._build_physical_panel(left)
        self._build_metrics_panel(right)
        self._build_log_panel(right)

    def _build_protocol_panel(self, parent):
        proto = ttk.LabelFrame(parent, text=" 📊 §4.3 Selective Repeat ARQ - sliding windows and channel occupancy ")
        proto.pack(fill="x", pady=(0, 5))

        self.proto_canvas = tk.Canvas(proto, height=PROTO_CANVAS_HEIGHT, bg=CANVAS_BG,
                                      highlightthickness=0)
        self.proto_canvas.pack(fill="x", padx=6, pady=(5, 2))
        self.proto_canvas.bind("<Configure>", lambda _event: self._redraw_last_state())

        legend = ttk.Frame(proto)
        legend.pack(fill="x", padx=8, pady=(0, 5))
        for text, color in [("unsent", STATE_COLORS["UNSENT"]), ("in flight", STATE_COLORS["SENT"]),
                            ("acked", STATE_COLORS["ACKED"]), ("nak", STATE_COLORS["NAK"]),
                            ("timed out", STATE_COLORS["TIMED_OUT"]), ("lost", STATE_COLORS["ABANDONED"]),
                            ("buffered", "#64B5F6")]:
            chip = tk.Canvas(legend, width=11, height=11, highlightthickness=1,
                             highlightbackground="#9E9E9E", bg=color)
            chip.pack(side="left", padx=(8, 3))
            ttk.Label(legend, text=text, style="Caption.TLabel").pack(side="left")

    def _build_framing_panel(self, parent):
        framing = ttk.LabelFrame(parent, text=" 🧩 §4.1 Framing - the frame as it sits on the wire ")
        framing.pack(fill="x", pady=(0, 5))

        self.framing_caption = ttk.Label(
            framing, style="Caption.TLabel",
            text="Waiting for the first frame…   "
                 "(amber = flag, magenta = stuffed bit, green = header/payload/CRC)")
        self.framing_caption.pack(anchor="w", padx=8, pady=(3, 0))

        self.framing_canvas = tk.Canvas(framing, height=FRAMING_CANVAS_HEIGHT, bg=SCOPE_BG,
                                        highlightthickness=0)
        self.framing_canvas.pack(fill="x", padx=8, pady=(2, 5))
        self.framing_canvas.bind("<Configure>", lambda _event: self._redraw_last_state())

    def _build_physical_panel(self, parent):
        phy = ttk.LabelFrame(parent, text=" 📡 §3.1/§3.2 Physical layer - transmitted vs received signal ")
        phy.pack(fill="both", expand=True)
        phy.columnconfigure(0, weight=1)
        phy.rowconfigure(1, weight=1)

        self.wave_caption = ttk.Label(phy, text="Waiting for the first frame…", style="Caption.TLabel")
        self.wave_caption.grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 0))

        waves = ttk.Frame(phy)
        waves.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 6))

        self.tx_wave_label = ttk.Label(waves, text="TX - clean signal leaving the transmitter",
                                       style="Caption.TLabel")
        self.tx_wave_label.pack(anchor="w")
        self.tx_wave_canvas = tk.Canvas(waves, height=WAVE_CANVAS_HEIGHT, bg=SCOPE_BG,
                                        highlightthickness=0)
        self.tx_wave_canvas.pack(fill="both", expand=True, pady=(1, 3))

        ttk.Label(waves, text="RX - signal recovered after the channel (AWGN / line distortion)",
                  style="Caption.TLabel").pack(anchor="w")
        self.rx_wave_canvas = tk.Canvas(waves, height=WAVE_CANVAS_HEIGHT, bg=SCOPE_BG,
                                        highlightthickness=0)
        self.rx_wave_canvas.pack(fill="both", expand=True, pady=(1, 0))
        self.rx_wave_canvas.bind("<Configure>", lambda _event: self._redraw_last_state())

        # Only 16-QAM has a quadrature component worth plotting, so this column is shown
        # and hidden according to the selected mode.
        self.const_column = ttk.Frame(phy)
        ttk.Label(self.const_column, text="§3.2 16-QAM constellation",
                  style="Caption.TLabel").pack(anchor="w")
        self.const_canvas = tk.Canvas(self.const_column, width=CONST_CANVAS_SIZE,
                                      height=CONST_CANVAS_SIZE, bg=SCOPE_BG,
                                      highlightthickness=0)
        self.const_canvas.pack()
        self.const_hint = ttk.Label(self.const_column, text="", style="Caption.TLabel")
        self.const_hint.pack(anchor="w")

    def _show_constellation(self, visible: bool):
        # Guarded because the drain loop asks for this many times per second, and
        # re-running the geometry manager on every tick makes the panel flicker.
        if visible == self._constellation_shown:
            return
        self._constellation_shown = visible
        if visible:
            self.const_column.grid(row=1, column=1, sticky="n", padx=(0, 8), pady=(0, 6))
        else:
            self.const_column.grid_remove()
            self.const_canvas.delete("all")
            self.const_hint.config(text="")

    def _build_metrics_panel(self, parent):
        metrics = ttk.LabelFrame(parent, text=" 📐 Live requirement metrics ")
        metrics.pack(fill="x", pady=(0, 5))
        metrics.columnconfigure(0, weight=1, uniform="metric")
        metrics.columnconfigure(1, weight=1, uniform="metric")

        self.metric_labels = {}
        cells = [
            ("channel", "§3.2 Channel"), ("security", "§5.3 Encryption"),
            ("framing", "§4.1 Frame on wire"), ("delimiter", "§4.1 Flag search"),
            ("linecode", "§2 Line code health"), ("fec", "§4.2 Hamming FEC"),
            ("crc", "§4.2 CRC / header drops"), ("arq", "§4.3 DATA tx / retx"),
            ("timers", "§4.3 Timeouts"), ("control", "§4.3 ACK / NAK"),
            ("throughput", "§5.1 Throughput"), ("goodput", "§5.1 Goodput"),
        ]
        for index, (key, caption) in enumerate(cells):
            cell = ttk.Frame(metrics, relief="groove", borderwidth=1)
            cell.grid(row=index // 2, column=index % 2, padx=3, pady=3, sticky="nsew")
            ttk.Label(cell, text=caption, style="Caption.TLabel").pack(anchor="w", padx=5, pady=(2, 0))
            value = ttk.Label(cell, text="-", style="Value.TLabel")
            value.pack(anchor="w", padx=5, pady=(0, 3))
            self.metric_labels[key] = value

        self.status_label = ttk.Label(metrics, text="Idle.", style="Caption.TLabel", wraplength=320)
        self.status_label.grid(row=len(cells) // 2, column=0, columnspan=2,
                               padx=6, pady=(2, 5), sticky="w")

    def _build_log_panel(self, parent):
        log_frame = ttk.LabelFrame(parent, text=" 💬 Protocol event log ")
        log_frame.pack(fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_frame, height=6, width=30, wrap="none", bg="#1C1C1C",
                                fg="#00E676", font=("Consolas", 8), yscrollcommand=log_scroll.set)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=5)
        log_scroll.config(command=self.log_text.yview)

    def _on_snr_change(self, _value=None):
        self.snr_label.config(text=f"{self.snr_var.get():.1f} dB")

    def update_channel_modes(self, event=None):
        """Only one impairment control applies at a time; the other is greyed out."""
        wired = self.channel_var.get() == "wired"
        self.mode_combo["values"] = ["B8ZS", "HDB3"] if wired else ["BPSK", "16-QAM"]
        self.mode_var.set("B8ZS" if wired else "BPSK")
        self.snr_scale.configure(state="disabled" if wired else "normal")
        self.snr_label.config(text="n/a (wired)" if wired else f"{self.snr_var.get():.1f} dB")
        self.ber_spin.configure(state="normal" if wired else "disabled")

    def load_message_file(self):
        path = filedialog.askopenfilename(
            title="Select a text file to transmit",
            filetypes=[("Text files", "*.txt *.md *.csv *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except OSError as exc:
            messagebox.showerror("File error", f"Could not read the file:\n{exc}")
            return
        self.msg_entry.delete(0, tk.END)
        self.msg_entry.insert(0, content.replace("\n", " ").strip())
        self.write_log(f"📂 [INPUT] Loaded {len(content)} characters from {path}")

    # ========================================================== live tab drawing

    def _redraw_last_state(self):
        """
        Re-renders every panel from the last state received.

        All of these canvases lay themselves out against their current pixel width, so after a
        resize the old drawing is stale: too short for the new width, or clipped by it. Any
        panel that draws must therefore be redrawable from stored state, not only on arrival.
        """
        if self._last_state:
            self._render_state(self._last_state)

    def _render_state(self, state):
        self.draw_protocol_state(state)

        wave = state.get("waveform")
        if wave:
            tx_q, rx_q = wave.get("tx_q"), wave.get("rx_q")
            self.draw_waveform(self.tx_wave_canvas, wave["tx_i"], TRACE_I_TX, tx_q, TRACE_Q_TX)
            self.draw_waveform(self.rx_wave_canvas, wave["rx_i"], TRACE_I_RX, rx_q, TRACE_Q_RX)
            self.wave_caption.config(text=state.get("waveform_label", ""))
            if tx_q and rx_q:
                self._show_constellation(True)
                self.draw_constellation(wave["tx_i"], tx_q, wave["rx_i"], rx_q)
            else:
                self._show_constellation(False)

        framing = state.get("framing") or {}
        if framing.get("bits"):
            total, flag = len(framing["bits"]), framing["flag_bits"]
            stuffed = len(framing["stuffed"])
            self.draw_framing_strip(framing["bits"], framing["stuffed"], flag)
            self.framing_caption.config(
                text=f"{total} bits on the wire = {flag}-bit flag + "
                     f"{total - 2 * flag}-bit body (including {stuffed} stuffed) + "
                     f"{flag}-bit flag   "
                     f"(amber = flag, magenta = stuffed bit, green = header/payload/CRC)")
            self.metric_labels["framing"].config(
                text=f"{total} b · {state['stats']['stuffed_bits']} stuffed total")

    def draw_protocol_state(self, state):
        c = self.proto_canvas
        c.delete("all")

        width = c.winfo_width() or 700
        height = c.winfo_height() or PROTO_CANVAS_HEIGHT
        num = max(state["num_frames"], 1)
        left, right = 78, 16
        avail = max(width - left - right, 120)

        pitch = min(44.0, avail / num)
        box_w = max(pitch - 4, 3)
        show_text = box_w >= 18

        # Derived from the live canvas height so the receive-window caption at the bottom
        # cannot fall off the edge when the panel is resized.
        box_h = 24
        y_send = 28
        y_recv = height - 50
        lane_fwd = y_send + box_h + 32
        lane_rev = y_recv - 22
        lane_x0, lane_x1 = left, left + avail

        base = state["send_base"]
        w_size = state["window_size"]
        states = state["frame_states"]
        timers = state["timers"]
        retries = state["retries"]
        recv_base = state["recv_base"]
        buffered = set(state["buffered"])

        c.create_text(6, y_send + box_h / 2, text="SENDER", anchor="w",
                      font=("Segoe UI", 8, "bold"), fill="#37474F")
        c.create_text(6, y_recv + box_h / 2, text="RECEIVER", anchor="w",
                      font=("Segoe UI", 8, "bold"), fill="#37474F")

        # Sender window: one box per frame, outlined while inside the sliding window.
        for i, st in enumerate(states):
            x1 = left + i * pitch
            x2 = x1 + box_w
            inside = base <= i < base + w_size
            c.create_rectangle(x1, y_send, x2, y_send + box_h,
                               fill=STATE_COLORS.get(st, "#E0E0E0"),
                               outline="#1565C0" if inside else "#BDBDBD",
                               width=2 if inside else 1)
            if show_text:
                c.create_text((x1 + x2) / 2, y_send + box_h / 2, text=str(i),
                              font=("Segoe UI", 8, "bold"))
                if st == "SENT" and i in timers:
                    c.create_text((x1 + x2) / 2, y_send + box_h + 8,
                                  text=f"⏱{timers[i]}/{state['timeout_limit']}",
                                  font=("Segoe UI", 7), fill="#616161")
                elif retries.get(i, 0) > 1:
                    c.create_text((x1 + x2) / 2, y_send + box_h + 8,
                                  text=f"×{retries[i]}", font=("Segoe UI", 7), fill="#C62828")

        if base < num:
            wx1 = left + base * pitch
            wx2 = left + min(base + w_size, num) * pitch - 4
            c.create_line(wx1, y_send - 8, wx2, y_send - 8, fill="#1565C0", width=2)
            c.create_text(wx1, y_send - 18, anchor="w", fill="#1565C0",
                          font=("Segoe UI", 8, "bold"),
                          text=f"send window  base={base}  W={w_size}")

        # Channel lanes: DATA travels left to right, ACK/NAK comes back.
        for y, label, arrow in ((lane_fwd, "FORWARD", "last"), (lane_rev, "← REVERSE", "first")):
            c.create_line(lane_x0, y, lane_x1, y, fill="#B0BEC5", width=1, dash=(4, 3), arrow=arrow)
            c.create_text(6, y, text=label, anchor="w", font=("Segoe UI", 7), fill="#78909C")
        c.create_text(lane_x1, lane_fwd - 13, text="DATA →", anchor="e",
                      font=("Segoe UI", 7), fill="#B0BEC5")
        c.create_text(lane_x1, lane_rev + 13, text="ACK / NAK", anchor="e",
                      font=("Segoe UI", 7), fill="#B0BEC5")

        for transit in state["transits"]:
            span = max(lane_x1 - lane_x0 - 38, 1)
            if transit["direction"] == "forward":
                x, y = lane_x0 + transit["progress"] * span, lane_fwd
            else:
                x, y = lane_x0 + (1.0 - transit["progress"]) * span, lane_rev
            color = TRANSIT_COLORS.get(transit["kind"], "#546E7A")
            c.create_rectangle(x, y - 8, x + 38, y + 8, fill=color, outline=color)
            c.create_text(x + 19, y, text=f"{transit['kind'][0]}{transit['seq']}",
                          fill="white", font=("Segoe UI", 7, "bold"))

        # Receiver window: delivered, buffered, or still awaited.
        for i in range(num):
            x1 = left + i * pitch
            x2 = x1 + box_w
            if i < recv_base:
                fill = STATE_COLORS["ACKED"]
            elif i in buffered:
                fill = "#64B5F6"
            elif recv_base <= i < recv_base + w_size:
                fill = "#FFFFFF"
            else:
                fill = "#EEEEEE"
            inside = recv_base <= i < recv_base + w_size
            c.create_rectangle(x1, y_recv, x2, y_recv + box_h, fill=fill,
                               outline="#00838F" if inside else "#BDBDBD",
                               width=2 if inside else 1)
            if show_text:
                c.create_text((x1 + x2) / 2, y_recv + box_h / 2, text=str(i),
                              font=("Segoe UI", 8, "bold"))

        if recv_base < num:
            rx1 = left + recv_base * pitch
            rx2 = left + min(recv_base + w_size, num) * pitch - 4
            c.create_line(rx1, y_recv + box_h + 8, rx2, y_recv + box_h + 8, fill="#00838F", width=2)
            c.create_text(rx1, y_recv + box_h + 17, anchor="w", fill="#00838F",
                          font=("Segoe UI", 8, "bold"),
                          text=f"receive window  base={recv_base}  buffered={sorted(buffered)[:8]}")

        c.create_text(width - 8, 10, anchor="e", fill="#455A64", font=("Segoe UI", 9, "bold"),
                      text=f"tick {state['tick']}")

    def draw_waveform(self, canvas: tk.Canvas, samples, color: str,
                      quadrature=None, quad_color: str = TRACE_Q_TX):
        """
        Plots the in-phase trace and, for 16-QAM, the quadrature trace beside it.

        Both traces share one amplitude scale, otherwise the eye would compare two
        differently normalised curves and read a difference that is not there.
        """
        canvas.delete("all")
        if not samples:
            return

        width = canvas.winfo_width() or 600
        height = canvas.winfo_height() or WAVE_CANVAS_HEIGHT
        mid = height / 2
        canvas.create_line(0, mid, width, mid, fill="#37474F", width=1)

        amplitude = max(
            max((abs(v) for v in samples), default=1.0),
            max((abs(v) for v in quadrature), default=0.0) if quadrature else 0.0,
        ) or 1.0

        step = max(1, len(samples) // max(width, 1))
        self._plot_trace(canvas, samples[::step], width, mid, amplitude, color, ())
        if quadrature:
            self._plot_trace(canvas, quadrature[::step], width, mid, amplitude,
                             quad_color, (3, 2))

        legend = "I / Q" if quadrature else "amplitude"
        canvas.create_text(width - 5, 9, anchor="e", fill="#78909C", font=("Consolas", 7),
                           text=f"{len(samples)} samples  peak {amplitude:.2f}  {legend}")

    @staticmethod
    def _plot_trace(canvas, points, width, mid, amplitude, color, dash):
        if len(points) < 2:
            return
        scale = mid - 7
        dx = width / (len(points) - 1)
        coords = []
        for index, value in enumerate(points):
            coords.extend((index * dx, mid - (value / amplitude) * scale))
        canvas.create_line(*coords, fill=color, width=1, dash=dash)

    def draw_framing_strip(self, bits, stuffed_positions, flag_bits):
        """
        Draws one frame bit by bit so the framing work is visible rather than implied.

        Tall ticks are ones and short ticks are zeros. The flags are amber, and every bit the
        stuffer inserted is magenta — which is what shows that the body genuinely cannot
        contain the flag pattern, the property the receiver's flag search depends on.
        """
        canvas = self.framing_canvas
        canvas.delete("all")
        if not bits:
            return

        width = canvas.winfo_width() or 600
        height = canvas.winfo_height() or FRAMING_CANVAS_HEIGHT
        total = len(bits)
        dx = width / total
        baseline = height - 10
        high, low = 8.0, baseline - 9

        stuffed = set(stuffed_positions)
        tick_width = max(1, int(dx))

        canvas.create_rectangle(0, 0, flag_bits * dx, height, fill="#2A2417", outline="")
        canvas.create_rectangle((total - flag_bits) * dx, 0, width, height,
                                fill="#2A2417", outline="")
        canvas.create_line(0, baseline, width, baseline, fill="#37474F")

        for index, bit in enumerate(bits):
            if index < flag_bits or index >= total - flag_bits:
                color = FLAG_COLOR
            elif index in stuffed:
                color = STUFF_COLOR
            else:
                color = BODY_COLOR
            canvas.create_line(index * dx, baseline, index * dx, high if bit else low,
                               fill=color, width=tick_width)

        canvas.create_text(2, height - 4, anchor="sw", fill="#FFD54F",
                           font=("Consolas", 7), text="FLAG")
        canvas.create_text(width - 2, height - 4, anchor="se", fill="#FFD54F",
                           font=("Consolas", 7), text="FLAG")

    def draw_constellation(self, tx_i, tx_q, rx_i, rx_q):
        """
        Scatter plot of the received symbols against the ideal 16-QAM grid.

        A dot is drawn red when the slicer would assign it to a different point than the
        one that was transmitted, so a symbol error is visible as a dot that has crossed a
        decision boundary rather than as an abstract counter.
        """
        canvas = self.const_canvas
        canvas.delete("all")
        size = CONST_CANVAS_SIZE
        span = 5.0

        def to_px(i_value, q_value):
            return (size / 2) * (1 + i_value / span), (size / 2) * (1 - q_value / span)

        for level in QAM16_DECISION_LEVELS:
            x, _ = to_px(level, 0)
            _, y = to_px(0, level)
            style = {"fill": "#37474F", "width": 1} if level == 0 else {"fill": "#2A3238",
                                                                       "dash": (2, 3)}
            canvas.create_line(x, 0, x, size, **style)
            canvas.create_line(0, y, size, y, **style)

        for point in PhysicalLayerWireless.constellation_points():
            x, y = to_px(point.real, point.imag)
            canvas.create_line(x - 3, y, x + 3, y, fill="#546E7A")
            canvas.create_line(x, y - 3, x, y + 3, fill="#546E7A")

        errors = 0
        pairs = list(zip(tx_i, tx_q, rx_i, rx_q))[:400]
        for ti, tq, ri, rq in pairs:
            wrong = (PhysicalLayerWireless._quantize(ri) != PhysicalLayerWireless._quantize(ti)
                     or PhysicalLayerWireless._quantize(rq) != PhysicalLayerWireless._quantize(tq))
            errors += wrong
            x, y = to_px(ri, rq)
            color = "#EF5350" if wrong else "#00E676"
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")

        self.const_hint.config(
            text=f"{len(pairs)} symbols · {errors} sliced to the wrong point")

    def write_log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    # ========================================================= live tab execution

    def start_live_simulation(self):
        if self.live_thread and self.live_thread.is_alive():
            return

        raw_msg = self.msg_entry.get()
        if not raw_msg:
            messagebox.showwarning("Empty message", "Enter a message or load a file first.")
            return

        key = self.key_entry.get() or "DefaultKey"
        channel = self.channel_var.get()
        mode = self.mode_var.get()
        snr = float(self.snr_var.get())
        window = int(self.w_spin.get())
        timeout = int(self.timeout_spin.get())
        prop = int(self.prop_spin.get())
        chunk = int(self.chunk_spin.get())
        retries = int(self.retry_spin.get())
        tick_delay = {"Slow": 0.30, "Normal": 0.12, "Fast": 0.04, "Instant": 0.0}[self.speed_combo.get()]
        try:
            error_rate = float(self.ber_spin.get())
        except ValueError:
            messagebox.showerror("Invalid input", "The wired BER must be a number.")
            return

        self.log_text.delete("1.0", tk.END)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Running…", foreground="#EF6C00")

        # Section 5.3: the plaintext is encrypted before it ever reaches the data link layer.
        plaintext = raw_msg.encode("utf-8")
        ciphertext = SecurityLayer.encrypt_xor_rotational(plaintext, key)
        self.write_log(f"🔑 [§5.3] {len(plaintext)} B plaintext encrypted to {len(ciphertext)} B "
                       f"before framing: {ciphertext[:10].hex()}…")
        if timeout <= 2 * prop:
            self.write_log(f"⚠️  [CONFIG] Timeout {timeout} ≤ round trip {2 * prop}; "
                           f"premature timeouts expected.")

        self.live_sim = SelectiveRepeatSimulation(
            encrypted_message=ciphertext,
            channel_type=channel,
            mode=mode,
            snr_db=snr,
            window_size=window,
            timeout_limit=timeout,
            chunk_size=chunk,
            prop_delay=prop,
            tick_delay=tick_delay,
            error_rate=error_rate,
            max_retries=retries,
            verbose=False,
            seed=None,
        )

        sample_bits = len(Frame(0, "DATA", ciphertext[:chunk]).to_bits())
        self.metric_labels["framing"].config(text=f"{sample_bits} bits / {chunk} B")
        self.metric_labels["security"].config(text=f"XOR rot · {len(ciphertext)} B")
        self.metric_labels["channel"].config(
            text=f"{channel} · {mode} · "
                 + (f"BER {error_rate:.3f}" if channel == "wired" else f"{snr:.1f} dB Eb/N0"))

        quadrature = mode == "16-QAM"
        self.tx_wave_label.config(
            text="TX - clean signal leaving the transmitter"
                 + (" (I solid, Q dashed)" if quadrature else ""))
        if not quadrature:
            self._show_constellation(False)

        self.live_queue = queue.Queue()
        max_ticks = max(400, self.live_sim.num_frames * 40)
        self.live_thread = threading.Thread(
            target=self._live_worker, args=(self.live_sim, key, plaintext, max_ticks), daemon=True)
        self.live_thread.start()
        self.root.after(40, self._drain_live_queue)

    def _live_worker(self, sim, key, plaintext, max_ticks):
        try:
            payload, throughput, goodput = sim.run(max_ticks=max_ticks,
                                                   progress_callback=self.live_queue.put)
            self.live_queue.put({"__done__": True, "payload": payload, "key": key,
                                 "plaintext": plaintext, "throughput": throughput,
                                 "goodput": goodput})
        except Exception:
            self.live_queue.put({"__error__": traceback.format_exc()})

    def stop_live_simulation(self):
        if self.live_sim:
            self.live_sim.request_stop()
            self.status_label.config(text="Stopping…", foreground="#EF6C00")

    def _drain_live_queue(self):
        latest, done, error, lines = None, None, None, []
        while True:
            try:
                item = self.live_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("__done__"):
                done = item
            elif item.get("__error__"):
                error = item["__error__"]
            else:
                latest = item
                lines.extend(item.get("log") or [])

        for line in lines:
            if line.strip():
                self.write_log(line)

        if latest:
            self._last_state = latest
            self._render_state(latest)
            self._update_metrics(latest["stats"])

        if error:
            self._finish_live(f"Error: {error.splitlines()[-1]}", "#C62828")
            messagebox.showerror("Simulation error", error)
            return

        if done:
            self._finalize_live(done)
            return

        self.root.after(40, self._drain_live_queue)

    def _update_metrics(self, stats: dict):
        self.metric_labels["fec"].config(text=f"{stats['hamming_corrections']} bits fixed")
        self.metric_labels["crc"].config(
            text=f"{stats['crc_drops']} CRC · {stats['header_drops']} hdr")
        self.metric_labels["arq"].config(
            text=f"{stats['data_transmissions']} / {stats['retransmissions']}")
        self.metric_labels["timers"].config(text=f"{stats['timeouts']} expired")
        self.metric_labels["control"].config(
            text=f"A {stats['acks_received']}/{stats['acks_sent']} · "
                 f"N {stats['naks_received']}/{stats['naks_sent']} · "
                 f"{stats['control_distrusted']} suspect")
        self.metric_labels["delimiter"].config(
            text=f"{stats['frames_delimited']} found · {stats['resyncs']} resync")
        self.metric_labels["throughput"].config(text=f"{stats['throughput']:.1f} b/tick")
        self.metric_labels["goodput"].config(
            text=f"{stats['goodput']:.1f} b/tick ({stats['efficiency']:.0%})")

        if self.channel_var.get() == "wired":
            self.metric_labels["linecode"].config(
                text=f"DC {stats['dc_balance']:+d} · gap {stats['longest_zero_run']}")
        else:
            self.metric_labels["linecode"].config(text="n/a (wireless)")

    def _finalize_live(self, done: dict):
        sim = self.live_sim
        recovered = SecurityLayer.decrypt_xor_rotational(done["payload"], done["key"])

        self.write_log("")
        self.write_log("=" * 60)
        if sim.success and recovered == done["plaintext"]:
            self.write_log("✅ TRANSMISSION COMPLETE - payload recovered bit-exactly")
            try:
                self.write_log(f"🔓 [§5.3] Decrypted: {recovered.decode('utf-8')}")
            except UnicodeDecodeError:
                self.write_log(f"🔓 [§5.3] Decrypted: {recovered!r}")
            self._finish_live(
                f"Delivered {sim.stats.frames_delivered}/{sim.num_frames} frames in "
                f"{sim.stats.ticks} ticks · efficiency {sim.stats.efficiency:.1%}", "#2E7D32")
        else:
            missing = sorted(set(range(sim.num_frames)) - set(sim.received_payloads))
            self.write_log("⛔ TRANSMISSION INCOMPLETE - the payload cannot be reconstructed")
            self.write_log(f"⛔ Frames never delivered: {missing}")
            self.write_log("⛔ Raise the SNR, enlarge the timeout, or allow more retries.")
            self._finish_live(
                f"FAILED · {len(missing)} frame(s) lost · delivered "
                f"{sim.stats.frames_delivered}/{sim.num_frames}", "#C62828")
        self.write_log("=" * 60)

    def _finish_live(self, status: str, color: str):
        self.status_label.config(text=status, foreground=color)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ============================================================== terminal tab

    def setup_terminal_tab(self, parent):
        control = ttk.LabelFrame(parent, text=" ⚙️ Single-run configuration ")
        control.pack(fill="x", padx=10, pady=8)
        control.columnconfigure(1, weight=1)

        ttk.Label(control, text="Message:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        self.term_msg_entry = ttk.Entry(control)
        self.term_msg_entry.insert(0, "Test message for terminal simulation mode")
        self.term_msg_entry.grid(row=0, column=1, columnspan=5, padx=5, pady=4, sticky="ew")

        ttk.Label(control, text="Key:").grid(row=1, column=0, padx=5, pady=4, sticky="w")
        self.term_key_entry = ttk.Entry(control, width=22)
        self.term_key_entry.insert(0, "TerminalTestKey")
        self.term_key_entry.grid(row=1, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(control, text="Channel:").grid(row=1, column=2, padx=5, pady=4, sticky="e")
        self.term_channel = tk.StringVar(value="wireless")
        term_channel_combo = ttk.Combobox(control, textvariable=self.term_channel,
                                          values=["wired", "wireless"], width=10, state="readonly")
        term_channel_combo.grid(row=1, column=3, padx=5, pady=4, sticky="w")

        ttk.Label(control, text="Mode:").grid(row=1, column=4, padx=5, pady=4, sticky="e")
        self.term_mode = tk.StringVar(value="BPSK")
        self.term_mode_combo = ttk.Combobox(control, textvariable=self.term_mode,
                                            values=["BPSK", "16-QAM"], width=10, state="readonly")
        self.term_mode_combo.grid(row=1, column=5, padx=5, pady=4, sticky="w")

        def sync_term_modes(_event=None):
            wired = self.term_channel.get() == "wired"
            self.term_mode_combo["values"] = ["B8ZS", "HDB3"] if wired else ["BPSK", "16-QAM"]
            self.term_mode.set("B8ZS" if wired else "BPSK")
        term_channel_combo.bind("<<ComboboxSelected>>", sync_term_modes)

        ttk.Label(control, text="Window:").grid(row=2, column=0, padx=5, pady=4, sticky="w")
        self.term_w_spin = ttk.Spinbox(control, from_=1, to=8, width=5, state="readonly")
        self.term_w_spin.set(4)
        self.term_w_spin.grid(row=2, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(control, text="SNR (dB):").grid(row=2, column=2, padx=5, pady=4, sticky="e")
        self.term_snr_spin = ttk.Spinbox(control, from_=0.0, to=20.0, increment=0.5, width=6)
        self.term_snr_spin.set(8.0)
        self.term_snr_spin.grid(row=2, column=3, padx=5, pady=4, sticky="w")

        self.run_term_btn = ttk.Button(control, text="▶️ Run Simulation",
                                       command=self.run_terminal_simulation)
        self.run_term_btn.grid(row=2, column=4, columnspan=2, padx=10, pady=4, sticky="ew")

        log_frame = ttk.LabelFrame(parent, text=" 📋 Full protocol trace ")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")
        self.term_log_text = tk.Text(log_frame, height=22, bg="#1C1C1C", fg="#00E676",
                                     font=("Consolas", 9), yscrollcommand=scrollbar.set)
        self.term_log_text.pack(fill="both", expand=True, padx=8, pady=6)
        scrollbar.config(command=self.term_log_text.yview)

    def write_term_log(self, text: str):
        self.term_log_text.insert(tk.END, text + "\n")
        self.term_log_text.see(tk.END)

    def run_terminal_simulation(self):
        if self.term_thread and self.term_thread.is_alive():
            return

        msg = self.term_msg_entry.get() or "Default terminal test message"
        key = self.term_key_entry.get() or "DefaultKey"
        window = int(self.term_w_spin.get())
        snr = float(self.term_snr_spin.get())
        channel = self.term_channel.get()
        mode = self.term_mode.get()

        self.term_log_text.delete("1.0", tk.END)
        self.run_term_btn.config(state="disabled")

        self.write_term_log("=" * 78)
        self.write_term_log("Terminal Simulation - Secure End-to-End Transmission".center(78))
        self.write_term_log("=" * 78)
        self.write_term_log(f"📝 Message: {msg}")
        self.write_term_log(f"⚙️  {channel} / {mode} | SNR {snr} dB | window {window}")

        plaintext = msg.encode("utf-8")
        ciphertext = SecurityLayer.encrypt_xor_rotational(plaintext, key)
        self.write_term_log(f"🔑 [§5.3] Encrypted {len(plaintext)} B -> {len(ciphertext)} B: {ciphertext.hex()}")
        self.write_term_log("")

        sim = SelectiveRepeatSimulation(
            encrypted_message=ciphertext, channel_type=channel, mode=mode, snr_db=snr,
            window_size=window, timeout_limit=6, chunk_size=8, prop_delay=2,
            verbose=False, seed=None,
        )

        self.term_queue = queue.Queue()

        def worker():
            try:
                payload, _, _ = sim.run(max_ticks=max(400, sim.num_frames * 40),
                                        progress_callback=self.term_queue.put)
                self.term_queue.put({"__done__": True, "payload": payload})
            except Exception:
                self.term_queue.put({"__error__": traceback.format_exc()})

        self.term_thread = threading.Thread(target=worker, daemon=True)
        self.term_thread.start()
        self.root.after(40, lambda: self._drain_term_queue(sim, plaintext, key))

    def _drain_term_queue(self, sim, plaintext, key):
        done, error, lines = None, None, []
        while True:
            try:
                item = self.term_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("__done__"):
                done = item
            elif item.get("__error__"):
                error = item["__error__"]
            else:
                lines.extend(item.get("log") or [])

        for line in lines:
            if line.strip():
                self.write_term_log(line)

        if error:
            self.write_term_log(error)
            self.run_term_btn.config(state="normal")
            return

        if done:
            recovered = SecurityLayer.decrypt_xor_rotational(done["payload"], key)
            self.write_term_log("")
            self.write_term_log(f"🔓 [§5.3] Decrypted: {recovered.decode('utf-8', errors='replace')}")
            self.write_term_log(f"✅ Bit-exact match: {recovered == plaintext}")
            self.run_term_btn.config(state="normal")
            return

        self.root.after(40, lambda: self._drain_term_queue(sim, plaintext, key))

    # =========================================================== performance tab

    def setup_performance_tab(self, parent):
        control = ttk.LabelFrame(parent, text=" ⚙️ §5.1 Throughput / Goodput sweep ")
        control.pack(fill="x", padx=10, pady=8)

        if not HAS_MATPLOTLIB or PerformanceAnalyzer is None:
            ttk.Label(parent, text="⚠️ Matplotlib is required for this feature.",
                      foreground="red").pack(pady=10)
            ttk.Label(parent, text="Install with: python -m pip install -r requirements.txt").pack(pady=5)
            return

        self.perf_axis_label = ttk.Label(control, text="Eb/N0 from", width=12)
        self.perf_axis_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.perf_snr_min = ttk.Spinbox(control, from_=-2.0, to=15.0, increment=0.5, width=6)
        self.perf_snr_min.set(0.0)
        self.perf_snr_min.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(control, text="to").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.perf_snr_max = ttk.Spinbox(control, from_=0.0, to=25.0, increment=0.5, width=6)
        self.perf_snr_max.set(12.0)
        self.perf_snr_max.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(control, text="step").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.perf_snr_step = ttk.Spinbox(control, from_=0.005, to=5.0, increment=0.5, width=6)
        self.perf_snr_step.set(1.5)
        self.perf_snr_step.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        ttk.Label(control, text="Window sizes:").grid(row=0, column=6, padx=5, pady=5, sticky="e")
        self.perf_windows = ttk.Entry(control, width=16)
        self.perf_windows.insert(0, "1, 2, 4, 8")
        self.perf_windows.grid(row=0, column=7, padx=5, pady=5, sticky="w")

        ttk.Label(control, text="Trials/point:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.perf_trials = ttk.Spinbox(control, from_=1, to=10, width=6, state="readonly")
        self.perf_trials.set(3)
        self.perf_trials.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(control, text="Channel (§2):").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.perf_channel = tk.StringVar(value="wireless")
        self.perf_channel_combo = ttk.Combobox(control, textvariable=self.perf_channel,
                                               values=["wired", "wireless"], width=10,
                                               state="readonly")
        self.perf_channel_combo.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.perf_channel_combo.bind("<<ComboboxSelected>>", self._on_perf_channel_change)

        ttk.Label(control, text="Mode:").grid(row=1, column=4, padx=5, pady=5, sticky="e")
        self.perf_mode = tk.StringVar(value="BPSK")
        self.perf_mode_combo = ttk.Combobox(control, textvariable=self.perf_mode,
                                            values=["BPSK", "16-QAM"], width=10, state="readonly")
        self.perf_mode_combo.grid(row=1, column=5, padx=5, pady=5, sticky="w")

        self.run_perf_btn = ttk.Button(control, text="📊 Run Analysis",
                                        command=self.run_performance_analysis)
        self.run_perf_btn.grid(row=1, column=6, columnspan=2, padx=10, pady=5, sticky="ew")

        self.perf_status = ttk.Label(
            parent, wraplength=1100,
            text="Ready. Throughput counts every bit placed on the channel; "
                 "goodput counts only payload delivered to the destination.",
            foreground="#1565C0")
        self.perf_status.pack(pady=4, padx=10, anchor="w")

        self.perf_frame = ttk.Frame(parent)
        self.perf_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.perf_queue: queue.Queue = queue.Queue()
        self.perf_thread = None

    def _on_perf_channel_change(self, event=None):
        """The swept axis is Eb/N0 on radio but a per-symbol error rate on copper."""
        wired = self.perf_channel.get() == "wired"
        self.perf_mode_combo["values"] = ["B8ZS", "HDB3"] if wired else ["BPSK", "16-QAM"]
        self.perf_mode.set("B8ZS" if wired else "BPSK")
        self.perf_axis_label.config(text="BER from" if wired else "Eb/N0 from")
        defaults = (0.0, 0.06, 0.01) if wired else (0.0, 12.0, 1.5)
        for spin, value in zip((self.perf_snr_min, self.perf_snr_max, self.perf_snr_step), defaults):
            spin.delete(0, tk.END)
            spin.insert(0, str(value))

    def run_performance_analysis(self):
        if self.perf_thread and self.perf_thread.is_alive():
            return
        channel = self.perf_channel.get()
        wired = channel == "wired"
        try:
            x_min = float(self.perf_snr_min.get())
            x_max = float(self.perf_snr_max.get())
            x_step = max(float(self.perf_snr_step.get()), 0.001 if wired else 0.1)
            windows = [int(x.strip()) for x in self.perf_windows.get().split(",") if x.strip()]
            trials = int(self.perf_trials.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", f"Could not read the parameters:\n{exc}")
            return

        if not windows:
            messagebox.showerror("Invalid input", "Provide at least one window size.")
            return

        if wired and not (0.0 <= x_min <= x_max <= 1.0):
            messagebox.showerror("Invalid input",
                                 "A wired sweep varies a per-symbol error rate, so the range "
                                 "must lie between 0 and 1.")
            return

        x_values = []
        value = x_min
        while value <= x_max + 1e-9:
            x_values.append(round(value, 4))
            value += x_step

        mode = self.perf_mode.get()
        self.run_perf_btn.config(state="disabled")
        self.perf_status.config(
            text=f"Running {len(windows) * len(x_values) * trials} {mode} simulations "
                 f"over the {channel} channel…",
            foreground="#EF6C00")
        self.perf_queue = queue.Queue()

        def worker():
            try:
                results = PerformanceAnalyzer.sweep(
                    x_values, windows, channel_type=channel, mode=mode, trials=trials,
                    progress=lambda text: self.perf_queue.put({"progress": text}))
                self.perf_queue.put({"__done__": True, "results": results, "x": x_values,
                                     "windows": windows, "mode": mode, "channel": channel})
            except Exception:
                self.perf_queue.put({"__error__": traceback.format_exc()})

        self.perf_thread = threading.Thread(target=worker, daemon=True)
        self.perf_thread.start()
        self.root.after(80, self._drain_perf_queue)

    def _drain_perf_queue(self):
        done, error, last_progress = None, None, None
        while True:
            try:
                item = self.perf_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("__done__"):
                done = item
            elif item.get("__error__"):
                error = item["__error__"]
            else:
                last_progress = item.get("progress")

        if last_progress:
            self.perf_status.config(text=last_progress, foreground="#EF6C00")

        if error:
            self.perf_status.config(text=f"❌ {error.splitlines()[-1]}", foreground="#C62828")
            self.run_perf_btn.config(state="normal")
            messagebox.showerror("Analysis error", error)
            return

        if done:
            fig = PerformanceAnalyzer.build_figure(done["x"], done["windows"],
                                                   done["results"], done["channel"], done["mode"],
                                                   figsize=(11.5, 4.4))
            for widget in self.perf_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.perf_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
            self.perf_status.config(
                text="✅ Analysis complete. Goodput saturates once the window covers the "
                     "round trip; the gap to throughput is protocol overhead.",
                foreground="#2E7D32")
            self.run_perf_btn.config(state="normal")
            return

        self.root.after(80, self._drain_perf_queue)


# Legacy class name for backward compatibility
class RealTimeGUI(UnifiedSimulatorGUI):
    pass
