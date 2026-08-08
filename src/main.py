"""Main entry point for the simulator."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk

try:
    from .security import SecurityLayer
    from .arq_protocol import SelectiveRepeatSimulation
except ImportError:  # pragma: no cover - fallback for direct execution
    from security import SecurityLayer
    from arq_protocol import SelectiveRepeatSimulation

try:
    from .performance_analysis import PerformanceAnalyzer
except Exception:  # pragma: no cover - optional dependency path
    PerformanceAnalyzer = None

try:
    from .gui import RealTimeGUI
except Exception:  # pragma: no cover - optional dependency path
    RealTimeGUI = None


def build_simulation(
    message: str,
    key: str,
    channel_type: str = "wireless",
    mode: str = "BPSK",
    snr_db: float = 8.0,
    window_size: int = 4,
    timeout_limit: int = 6,
    chunk_size: int = 8,
    prop_delay: int = 2,
    verbose: bool = True,
    seed: int | None = None,
):
    """Encrypts the message (section 5.3) and wires it into a ready-to-run simulation."""
    encrypted_bytes = SecurityLayer.encrypt_xor_rotational(message.encode("utf-8"), key)
    sim = SelectiveRepeatSimulation(
        encrypted_message=encrypted_bytes,
        channel_type=channel_type,
        mode=mode,
        snr_db=snr_db,
        window_size=window_size,
        timeout_limit=timeout_limit,
        chunk_size=chunk_size,
        prop_delay=prop_delay,
        verbose=verbose,
        seed=seed,
    )
    return encrypted_bytes, sim


def run_headless(args: argparse.Namespace) -> None:
    """Runs one transmission in the terminal and reports whether the payload survived."""
    _, sim = build_simulation(
        message=args.message,
        key=args.key,
        channel_type=args.channel,
        mode=args.mode,
        snr_db=args.snr,
        window_size=args.window,
        timeout_limit=args.timeout,
        seed=args.seed,
    )
    payload, throughput, goodput = sim.run(max_ticks=max(400, sim.num_frames * 40))
    recovered = SecurityLayer.decrypt_xor_rotational(payload, args.key)

    print(f"\nDecrypted output: {recovered.decode('utf-8', errors='replace')}")
    print(f"Bit-exact match : {recovered == args.message.encode('utf-8')}")
    print(f"Throughput      : {throughput:.2f} bits/tick")
    print(f"Goodput         : {goodput:.2f} bits/tick")


def launch_gui() -> None:
    if RealTimeGUI is None:
        print("GUI support is unavailable in this environment.")
        return
    print("Loading GUI...")
    root = tk.Tk()
    RealTimeGUI(root)
    root.mainloop()


def _use_utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot encode the log emoji."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main() -> None:
    _use_utf8_console()
    parser = argparse.ArgumentParser(description="End-to-end physical + data link layer simulator.")
    parser.add_argument("--analyze", action="store_true",
                        help="run the section 5.1 Throughput/Goodput sweep and save the chart")
    parser.add_argument("--terminal", action="store_true",
                        help="run a single transmission in the terminal instead of the GUI")
    parser.add_argument("--message", default="End-to-end secure transmission test message.")
    parser.add_argument("--key", default="SecureXORKey")
    parser.add_argument("--channel", default="wireless", choices=["wired", "wireless"])
    parser.add_argument("--mode", default="BPSK", choices=["BPSK", "16-QAM", "B8ZS", "HDB3"])
    parser.add_argument("--snr", type=float, default=8.0)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.analyze:
        if PerformanceAnalyzer is None:
            print("Performance analysis is unavailable (matplotlib missing).")
            return
        PerformanceAnalyzer.run_analysis()
        return

    if args.terminal:
        run_headless(args)
        return

    print("\n" + "=" * 80)
    print("  Launching Unified Network Simulator  ".center(80, "#"))
    print("=" * 80)
    print("  All three simulation modes are available in the GUI:")
    print("  - 🎬 Live Transmission with sliding windows, channel occupancy and waveforms")
    print("  - 💻 Terminal Simulation for a single annotated run")
    print("  - 📈 Performance Analysis for the SNR / window size sweep")
    print("\n  Headless options: python run.py --terminal | python run.py --analyze")
    print("=" * 80 + "\n")
    launch_gui()


if __name__ == "__main__":
    main()
