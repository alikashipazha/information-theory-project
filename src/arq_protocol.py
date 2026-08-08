"""
Selective Repeat ARQ (Specification section 4.3).

The simulation is event driven: a frame handed to the physical layer occupies the channel
for a configurable propagation delay before the receiver sees it, and the resulting ACK or
NAK travels back over the same noisy medium. Nothing is acknowledged instantaneously, so
the per-frame timers, the sliding window and the retransmission logic all do real work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .data_link_layer import Frame, FrameDelimiter
    from .physical_layer import ChannelResult, CommunicationChannel, PhysicalLayerWired
except ImportError:  # pragma: no cover - allows direct execution from the src directory
    from data_link_layer import Frame, FrameDelimiter
    from physical_layer import ChannelResult, CommunicationChannel, PhysicalLayerWired


# Width of the sequence number field on the wire (8 bits in the frame header).
SEQ_SPACE = 256

UNSENT = "UNSENT"
IN_FLIGHT = "SENT"
ACKED = "ACKED"
NAKED = "NAK"
TIMED_OUT = "TIMED_OUT"
ABANDONED = "ABANDONED"


@dataclass
class Transit:
    """A frame currently occupying the channel between the two endpoints."""

    seq: int
    kind: str            # 'DATA', 'ACK' or 'NAK'
    direction: str       # 'forward' or 'reverse'
    bits: List[int]      # bits as they will be seen by the far end (noise already applied)
    depart_tick: int
    arrival_tick: int
    attempt: int = 1

    def progress(self, tick: int) -> float:
        span = max(self.arrival_tick - self.depart_tick, 1)
        return min(max((tick - self.depart_tick) / span, 0.0), 1.0)


@dataclass
class SimulationStats:
    ticks: int = 0
    data_transmissions: int = 0
    retransmissions: int = 0
    timeouts: int = 0
    acks_sent: int = 0
    naks_sent: int = 0
    acks_received: int = 0
    naks_received: int = 0
    control_lost: int = 0
    header_drops: int = 0
    crc_drops: int = 0
    hamming_corrections: int = 0
    frames_delivered: int = 0
    total_bits_sent: int = 0
    delivered_payload_bits: int = 0
    throughput: float = 0.0
    goodput: float = 0.0

    # Section 4.1 framing
    frames_delimited: int = 0
    resyncs: int = 0
    discarded_bits: int = 0
    flag_losses: int = 0
    stuffed_bits: int = 0
    control_distrusted: int = 0

    # Section 2a line-code health, wired channel only
    dc_balance: int = 0
    longest_zero_run: int = 0

    @property
    def efficiency(self) -> float:
        return self.goodput / self.throughput if self.throughput else 0.0

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["efficiency"] = self.efficiency
        return data


class SelectiveRepeatSimulation:
    def __init__(
        self,
        encrypted_message: bytes,
        channel_type: str = "wireless",
        mode: str = "BPSK",
        snr_db: Optional[float] = 8.0,
        window_size: int = 3,
        timeout_limit: int = 3,
        verbose: bool = True,
        seed: Optional[int] = None,
        drop_ack_for_sequences: Optional[Set[int]] = None,
        chunk_size: int = 4,
        max_retries: int = 5,
        prop_delay: int = 1,
        tick_delay: float = 0.0,
        error_rate: Optional[float] = None,
    ):
        # Fail fast: an invalid channel/mode pair used to fall through to an error-free
        # channel and silently report a flawless run.
        CommunicationChannel.validate_link(channel_type, mode, snr_db)

        self.channel_type = channel_type
        self.mode = mode
        self.snr_db = snr_db
        self.window_size = window_size
        self.timeout_limit = timeout_limit
        self.verbose = verbose
        self.seed = seed
        self.max_retries = max_retries
        self.prop_delay = max(1, prop_delay)
        self.tick_delay = tick_delay
        self.error_rate = error_rate

        # Each endpoint delimits the stream it receives, so framing errors are visible on
        # both directions of the link.
        self.rx_delimiter = FrameDelimiter()
        self.tx_delimiter = FrameDelimiter()

        # One long-lived generator: every transmission draws fresh noise, yet the whole run
        # stays reproducible for a given seed.
        self.rng = np.random.default_rng(seed)

        self.chunk_size = chunk_size
        self.payloads = [encrypted_message[i:i + chunk_size] for i in range(0, len(encrypted_message), chunk_size)]
        self.num_frames = len(self.payloads)

        # Sender state
        self.frame_states: List[str] = [UNSENT] * self.num_frames
        self.timers: Dict[int, int] = {}
        self.retry_attempts: Dict[int, int] = {}
        self.retransmit_queue: List[int] = []
        self.send_base = 0

        # Receiver state
        self.recv_base = 0
        self.received_payloads: Dict[int, bytes] = {}
        self.nak_sent_for: Set[int] = set()

        # Channel state
        self.forward_transits: List[Transit] = []
        self.reverse_transits: List[Transit] = []

        self.stats = SimulationStats()
        self.total_ticks = 0
        self.history: List[str] = []
        self.success = False
        self.failed_frames: Set[int] = set()
        self.last_waveform: Optional[dict] = None
        self.last_waveform_label: str = ""
        self.last_frame_bits: List[int] = []
        self.last_frame_stuffed: List[int] = []
        self.last_frame_flag_bits: int = len(Frame.FLAG)
        self.progress_callback: Optional[Callable[[dict], None]] = None

        self.drop_ack_for_sequences = set(drop_ack_for_sequences or [])
        self.ack_drop_seen: Set[int] = set()

        self._stop_requested = False
        self._log_cursor = 0

    # Kept for backward compatibility with existing callers and tests.
    @property
    def total_transmissions(self) -> int:
        return self.stats.data_transmissions

    def request_stop(self) -> None:
        """Asks a running simulation to finish at the end of the current tick."""
        self._stop_requested = True

    # ------------------------------------------------------------------ logging

    def log(self, text: str):
        self.history.append(text)
        if self.verbose:
            try:
                print(text)
            except UnicodeEncodeError:
                # Legacy Windows consoles default to cp1252 and cannot render the status
                # emoji; degrade the console output rather than abort the simulation.
                print(text.encode("ascii", "replace").decode("ascii"))

    def _emit_progress(self, message: Optional[str] = None):
        if self.progress_callback is None:
            self._log_cursor = len(self.history)
            return

        new_lines = self.history[self._log_cursor:]
        self._log_cursor = len(self.history)

        self.progress_callback(
            {
                "tick": self.total_ticks,
                "send_base": self.send_base,
                "recv_base": self.recv_base,
                "window_size": self.window_size,
                "num_frames": self.num_frames,
                "frame_states": list(self.frame_states),
                "timers": dict(self.timers),
                "timeout_limit": self.timeout_limit,
                "retries": dict(self.retry_attempts),
                "buffered": sorted(self.received_payloads.keys()),
                "transits": [
                    {
                        "seq": t.seq,
                        "kind": t.kind,
                        "direction": t.direction,
                        "progress": t.progress(self.total_ticks),
                    }
                    for t in (self.forward_transits + self.reverse_transits)
                ],
                "waveform": self.last_waveform,
                "waveform_label": self.last_waveform_label,
                "framing": {
                    "bits": self.last_frame_bits,
                    "stuffed": self.last_frame_stuffed,
                    "flag_bits": self.last_frame_flag_bits,
                },
                "stats": self._snapshot_stats().as_dict(),
                "log": new_lines,
                "message": message,
            }
        )

    def _snapshot_stats(self) -> SimulationStats:
        self.stats.ticks = self.total_ticks
        ticks = max(self.total_ticks, 1)
        self.stats.throughput = self.stats.total_bits_sent / ticks
        self.stats.goodput = self.stats.delivered_payload_bits / ticks
        return self.stats

    # --------------------------------------------------------------- main loop

    def run(self, max_ticks: int = 2000, progress_callback: Optional[Callable[[dict], None]] = None) -> Tuple[bytes, float, float]:
        """Runs the simulation and returns (reconstructed_payload, throughput, goodput)."""
        self.progress_callback = progress_callback

        self.log("=" * 80)
        self.log("Selective Repeat ARQ Simulation Started".center(80))
        self.log(f"Frames: {self.num_frames} | Window Size: {self.window_size} | Timeout: {self.timeout_limit} ticks")
        self.log(f"Propagation delay: {self.prop_delay} tick(s) each way -> round trip {2 * self.prop_delay} ticks")
        self.log(f"Channel: {self.channel_type.upper()} | Mode: {self.mode}" + (f" | SNR: {self.snr_db} dB" if self.channel_type == "wireless" else ""))
        if self.timeout_limit <= 2 * self.prop_delay:
            self.log(f"⚠️  [CONFIG] Timeout ({self.timeout_limit}) is not larger than the round trip "
                     f"({2 * self.prop_delay}); premature timeouts are expected.")
        self.log("=" * 80)
        self._emit_progress(message="started")

        while self.total_ticks < max_ticks and not self._stop_requested:
            if self.send_base >= self.num_frames:
                break

            self.total_ticks += 1
            self.log(f"\n--- [Tick {self.total_ticks}] (Send window: {self.send_base}..{min(self.send_base + self.window_size - 1, self.num_frames - 1)} | Receive base: {self.recv_base}) ---")

            self._deliver_forward()
            self._deliver_reverse()
            self._expire_timers()
            self._serve_transmit_queue()

            self._emit_progress()
            if self.tick_delay:
                time.sleep(self.tick_delay)

        reconstructed = b"".join(self.received_payloads[i] for i in sorted(self.received_payloads))
        self.success = len(self.received_payloads) == self.num_frames

        stats = self._snapshot_stats()
        self._log_summary(stats)
        self._emit_progress(message="finished")

        return reconstructed, stats.throughput, stats.goodput

    def _log_summary(self, stats: SimulationStats):
        self.log("=" * 80)
        if self.success:
            self.log("Transmission Completed - all frames delivered".center(80))
        else:
            missing = sorted(set(range(self.num_frames)) - set(self.received_payloads))
            self.log("TRANSMISSION INCOMPLETE - payload is not recoverable".center(80))
            self.log(f"  ⛔ Frames never delivered: {missing}")
            if self._stop_requested:
                self.log("  ⛔ Reason: stopped by user.")
            elif self.failed_frames:
                self.log(f"  ⛔ Reason: retry budget of {self.max_retries} exhausted for {sorted(self.failed_frames)}.")
            else:
                self.log("  ⛔ Reason: tick budget exhausted before the window drained.")
        self.log("=" * 80)
        self.log("Statistics:")
        self.log(f"  - Ticks elapsed: {stats.ticks}")
        self.log(f"  - Frames delivered: {stats.frames_delivered}/{self.num_frames}")
        self.log(f"  - DATA transmissions: {stats.data_transmissions} (of which {stats.retransmissions} retransmissions)")
        self.log(f"  - Timeouts: {stats.timeouts} | ACKs: {stats.acks_sent} sent / {stats.acks_received} received | NAKs: {stats.naks_sent} sent / {stats.naks_received} received")
        self.log(f"  - Hamming single-bit corrections: {stats.hamming_corrections}")
        self.log(f"  - Frames dropped on CRC-32: {stats.crc_drops} | unidentifiable headers: {stats.header_drops} | control frames lost: {stats.control_lost}")
        self.log(f"  - Control frames discarded for an untrustworthy repaired header: {stats.control_distrusted}")
        self.log(f"  - Framing: {stats.frames_delimited} frames delimited by flag search | "
                 f"{stats.stuffed_bits} bits stuffed | {stats.resyncs} resynchronisations | "
                 f"{stats.discarded_bits} bits discarded")
        if self.channel_type == "wired":
            self.log(f"  - Line code health: cumulative DC balance {stats.dc_balance} | "
                     f"longest run without a pulse {stats.longest_zero_run}")
        self.log(f"  - Throughput: {stats.throughput:.2f} bits/tick (every bit placed on the channel)")
        self.log(f"  - Goodput: {stats.goodput:.2f} bits/tick (payload actually delivered upward)")
        self.log(f"  - Efficiency (Goodput/Throughput): {stats.efficiency:.2%}")
        self.log("=" * 80)

    # ------------------------------------------------------------ channel plumbing

    def _send_over_channel(self, bits: List[int], want_waveform: bool) -> ChannelResult:
        """Every frame, data or control, goes through here so all of them face the channel."""
        if want_waveform:
            return CommunicationChannel.transmit(
                bits, self.channel_type, self.mode, self.snr_db,
                return_waveform=True, rng=self.rng, error_rate=self.error_rate,
            )
        rx_bits = CommunicationChannel.transmit(
            bits, self.channel_type, self.mode, self.snr_db, rng=self.rng,
            error_rate=self.error_rate,
        )
        return ChannelResult(rx_bits=rx_bits)

    def _resolve_seq(self, wire_seq: int, base: int) -> Optional[int]:
        """
        Maps a modular wire sequence number back to an absolute frame index.

        Only indices within one window either side of the current base are plausible, and
        since SEQ_SPACE is far larger than 2*window_size that mapping is unambiguous.
        """
        low = max(0, base - self.window_size)
        for candidate in range(low, base + self.window_size + 1):
            if candidate % SEQ_SPACE == wire_seq:
                return candidate
        return None

    # ------------------------------------------------------------------- sender

    def _serve_transmit_queue(self):
        """Places at most one DATA frame per tick on the shared forward channel."""
        seq = self._next_frame_to_send()
        if seq is None:
            return
        self._transmit_data(seq)

    def _next_frame_to_send(self) -> Optional[int]:
        while self.retransmit_queue:
            seq = self.retransmit_queue[0]
            if self.frame_states[seq] in (ACKED, ABANDONED):
                self.retransmit_queue.pop(0)
                continue
            return self.retransmit_queue.pop(0)

        for seq in range(self.send_base, min(self.send_base + self.window_size, self.num_frames)):
            if self.frame_states[seq] == UNSENT:
                return seq
        return None

    def _transmit_data(self, seq: int):
        attempt = self.retry_attempts.get(seq, 0) + 1
        self.retry_attempts[seq] = attempt

        frame = Frame(seq_num=seq % SEQ_SPACE, frame_type="DATA", payload=self.payloads[seq])
        encoding = frame.encode()
        tx_bits = encoding.bits
        sent = self._send_over_channel(tx_bits, want_waveform=True)

        self.stats.total_bits_sent += len(tx_bits)
        self.stats.data_transmissions += 1
        self.stats.stuffed_bits += encoding.stuffed_count
        if attempt > 1:
            self.stats.retransmissions += 1

        if self.channel_type == "wired":
            # The two properties a bipolar line code exists to control (section 2a). DC is
            # cumulative because that is what physically accumulates on the line; the gap is
            # a worst case because one long gap is enough to lose the clock.
            self.stats.dc_balance += PhysicalLayerWired.dc_imbalance(sent.tx_waveform)
            self.stats.longest_zero_run = max(
                self.stats.longest_zero_run,
                PhysicalLayerWired.longest_zero_run(sent.tx_waveform),
            )

        self.last_waveform = {
            "tx_i": sent.tx_waveform,
            "rx_i": sent.rx_waveform,
            "tx_q": sent.tx_quadrature,
            "rx_q": sent.rx_quadrature,
        }
        self.last_frame_bits = list(tx_bits)
        self.last_frame_stuffed = list(encoding.stuffed_positions)
        self.last_frame_flag_bits = encoding.flag_bits
        self.last_waveform_label = (
            f"DATA frame {seq} - attempt {attempt} ({len(tx_bits)} bits, "
            f"{encoding.stuffed_count} stuffed)"
        )

        label = "SEND" if attempt == 1 else "RESEND"
        self.log(f"📤 [{label}] Frame {seq} on the {self.channel_type} channel via {self.mode} (attempt {attempt}).")

        self.forward_transits.append(
            Transit(seq=seq, kind="DATA", direction="forward", bits=sent.rx_bits,
                    depart_tick=self.total_ticks, arrival_tick=self.total_ticks + self.prop_delay,
                    attempt=attempt)
        )
        self.frame_states[seq] = IN_FLIGHT
        self.timers[seq] = 0

    def _expire_timers(self):
        for seq in sorted(self.timers):
            if self.frame_states[seq] != IN_FLIGHT:
                continue
            self.timers[seq] += 1
            if self.timers[seq] >= self.timeout_limit:
                self.stats.timeouts += 1
                self.log(f"⏰ [TIMEOUT] Frame {seq} timer expired after {self.timers[seq]} ticks with no acknowledgement.")
                self._schedule_retransmit(seq, "TIMEOUT")

    def _deliver_reverse(self):
        arrived, self.reverse_transits = self._split_arrivals(self.reverse_transits)
        for transit in arrived:
            # The sender has to delimit the reverse stream too; a control frame whose flag
            # was damaged simply never appears.
            frames = self.tx_delimiter.feed(transit.bits)
            if not frames:
                self.stats.control_lost += 1
                self.stats.flag_losses += 1
                self.log(f"💥 [CTRL-LOST] A {transit.kind} lost its framing in transit; "
                         f"the timer must recover it.")
                continue
            for frame_bits in frames:
                self._sender_process(frame_bits, transit.kind)

    def _sender_process(self, frame_bits: List[int], kind: str):
        result = Frame.decode(frame_bits)
        self.stats.hamming_corrections += result.corrected_bits

        if not (result.header_valid and result.payload_valid):
            self.stats.control_lost += 1
            self.log(f"💥 [CTRL-LOST] A {kind} was corrupted in transit ({result.error}); the timer must recover it.")
            return

        if result.header_corrected:
            # A control frame is nothing but a header, so a Hamming correction inside it is
            # not worth trusting: two damaged bits make the decoder "repair" an innocent third
            # one and hand back a header that never existed. The costs are lopsided — a
            # discarded ACK costs one timeout, whereas acting on a wrong sequence number marks
            # the wrong frame delivered and loses payload for good. This does not eliminate
            # the risk (a mis-correction can still land outside the header while the header
            # CRC-16 coincidentally agrees), it removes the one case we can actually detect.
            self.stats.control_lost += 1
            self.stats.control_distrusted += 1
            self.log(f"🧪 [CTRL-SUSPECT] A {kind} needed a Hamming repair inside its header; "
                     f"its sequence number is not trustworthy, so it is discarded.")
            return

        seq = self._resolve_seq(result.seq_num, self.send_base)
        if seq is None or seq >= self.num_frames:
            return

        if result.frame_type == "ACK":
            self._handle_ack(seq)
        elif result.frame_type == "NAK":
            self._handle_nak(seq)

    def _handle_ack(self, seq: int):
        self.stats.acks_received += 1
        if self.frame_states[seq] == ACKED:
            self.log(f"🔁 [ACK-DUP] Duplicate ACK for Frame {seq} ignored.")
            return

        self.log(f"✅ [ACK] Acknowledgement received for Frame {seq}.")
        self.frame_states[seq] = ACKED
        self.timers.pop(seq, None)
        self._advance_send_base()

    def _handle_nak(self, seq: int):
        self.stats.naks_received += 1
        if self.frame_states[seq] == ACKED:
            return
        self.log(f"⚡ [NAK] Negative acknowledgement received for Frame {seq}; scheduling a selective retransmission.")
        self._schedule_retransmit(seq, "NAK")

    def _schedule_retransmit(self, seq: int, reason: str):
        if self.frame_states[seq] in (ACKED, ABANDONED):
            return

        if self.retry_attempts.get(seq, 0) >= self.max_retries:
            self.frame_states[seq] = ABANDONED
            self.failed_frames.add(seq)
            self.timers.pop(seq, None)
            self.log(f"🛑 [LOST] Frame {seq} exhausted its {self.max_retries}-attempt budget and is reported as LOST to the upper layer.")
            self._advance_send_base()
            return

        self.frame_states[seq] = NAKED if reason == "NAK" else TIMED_OUT
        self.timers.pop(seq, None)
        if seq not in self.retransmit_queue:
            self.retransmit_queue.append(seq)

    def _advance_send_base(self):
        moved = False
        while self.send_base < self.num_frames and self.frame_states[self.send_base] in (ACKED, ABANDONED):
            self.send_base += 1
            moved = True
        if moved:
            self.log(f"🔄 [SLIDE] Sender window slid forward. New base: {self.send_base}")

    # ----------------------------------------------------------------- receiver

    def _deliver_forward(self):
        arrived, self.forward_transits = self._split_arrivals(self.forward_transits)
        for transit in arrived:
            self._receiver_accept_bits(transit)

    def _receiver_accept_bits(self, transit: Transit):
        """
        Hands the arriving bits to the frame delimiter rather than assuming a framed unit.

        This is where section 4.1 becomes real: the receiver is given a bit stream and has to
        locate the flags itself. If noise damaged a flag the delimiter finds no frame here and
        the bits are counted as lost to resynchronisation, which the sender's timer recovers
        from — exactly the failure mode that bit stuffing and flags are designed around.
        """
        before_found = self.rx_delimiter.frames_found
        before_discarded = self.rx_delimiter.discarded_bits

        for frame_bits in self.rx_delimiter.feed(transit.bits):
            self._receiver_process(frame_bits)

        self.stats.frames_delimited = self.rx_delimiter.frames_found
        self.stats.resyncs = self.rx_delimiter.resyncs
        self.stats.discarded_bits = self.rx_delimiter.discarded_bits

        if self.rx_delimiter.frames_found == before_found:
            lost = self.rx_delimiter.discarded_bits - before_discarded
            self.stats.flag_losses += 1
            self.log(f"🚧 [FRAMING] No flag pair found in the arriving bits for Frame "
                     f"{transit.seq}; {lost} bit(s) dropped while resynchronising.")

    def _split_arrivals(self, transits: List[Transit]) -> Tuple[List[Transit], List[Transit]]:
        arrived = [t for t in transits if t.arrival_tick <= self.total_ticks]
        pending = [t for t in transits if t.arrival_tick > self.total_ticks]
        return arrived, pending

    def _receiver_process(self, frame_bits: List[int]):
        result = Frame.decode(frame_bits)
        self.stats.hamming_corrections += result.corrected_bits
        if result.corrected_bits:
            self.log(f"🩹 [FEC] Hamming(12,8) repaired {result.corrected_bits} single-bit error(s) in the arriving frame.")

        if not result.header_valid:
            self.stats.header_drops += 1
            self.log(f"❌ [DROP] An arriving frame is unidentifiable ({result.error}); no NAK is possible, the sender must time out.")
            return

        seq = self._resolve_seq(result.seq_num, self.recv_base)
        if seq is None:
            self.stats.header_drops += 1
            return

        if not result.payload_valid:
            self.stats.crc_drops += 1
            self.log(f"❌ [CRC-32] Frame {seq} carries a residual burst error and is discarded.")
            self._send_control("NAK", seq)
            return

        if seq < self.recv_base:
            self.log(f"🔁 [DUP] Frame {seq} was already delivered; re-acknowledging.")
            self._send_control("ACK", seq)
            return

        if seq >= self.recv_base + self.window_size:
            self.log(f"🚫 [WINDOW] Frame {seq} falls outside the receive window; discarded.")
            return

        if seq not in self.received_payloads:
            self.received_payloads[seq] = result.payload
            self.log(f"📦 [BUFFER] Frame {seq} verified and buffered in the receive window.")

        self._send_control("ACK", seq)
        self._deliver_in_order()

        # Selective Repeat: an out-of-order arrival exposes the gap at recv_base.
        if seq > self.recv_base and self.recv_base not in self.nak_sent_for:
            self.nak_sent_for.add(self.recv_base)
            self.log(f"🔎 [GAP] Frame {self.recv_base} is missing while {seq} arrived; requesting it explicitly.")
            self._send_control("NAK", self.recv_base)

    def _deliver_in_order(self):
        delivered = []
        while self.recv_base in self.received_payloads:
            self.stats.delivered_payload_bits += len(self.received_payloads[self.recv_base]) * 8
            self.stats.frames_delivered += 1
            delivered.append(self.recv_base)
            self.recv_base += 1
        if delivered:
            self.nak_sent_for = {s for s in self.nak_sent_for if s >= self.recv_base}
            self.log(f"⬆️  [DELIVER] Frames {delivered} handed to the upper layer in order.")

    def _send_control(self, kind: str, seq: int):
        frame = Frame(seq_num=seq % SEQ_SPACE, frame_type=kind, payload=b"")
        tx_bits = frame.to_bits()
        self.stats.total_bits_sent += len(tx_bits)

        if kind == "ACK":
            self.stats.acks_sent += 1
            if seq in self.drop_ack_for_sequences and seq not in self.ack_drop_seen:
                self.ack_drop_seen.add(seq)
                self.stats.control_lost += 1
                self.log(f"📭 [ACK-LOST] ACK for Frame {seq} was lost on the reverse channel.")
                return
        else:
            self.stats.naks_sent += 1
            self.log(f"⚠️  [NAK] Sending NAK for Frame {seq} back to the sender.")

        sent = self._send_over_channel(tx_bits, want_waveform=False)
        self.reverse_transits.append(
            Transit(seq=seq, kind=kind, direction="reverse", bits=sent.rx_bits,
                    depart_tick=self.total_ticks, arrival_tick=self.total_ticks + self.prop_delay)
        )
