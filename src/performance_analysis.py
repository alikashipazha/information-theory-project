"""
Performance Analysis Module (Specification section 5.1).

Sweeps channel quality (Eb/N0 on the wireless channel, per-symbol error rate on the wired
one) against sliding-window size, then plots Throughput (every bit placed on the channel)
against Goodput (payload actually delivered to the destination). Both the command line and
the GUI call into this single implementation so the two never drift apart.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from .arq_protocol import SelectiveRepeatSimulation
except ImportError:  # pragma: no cover
    from arq_protocol import SelectiveRepeatSimulation


DEFAULT_MESSAGE = b"This is standard test data of a decent length to measure performance accurately."


class PerformanceAnalyzer:
    """Batch runner behind the Throughput/Goodput charts."""

    # The swept axis is not the same physical quantity on both channels, so every place that
    # sweeps, labels or reports it asks here instead of assuming Eb/N0.
    AXIS_LABELS = {
        "wireless": ("Eb/N0 (dB) - energy per information bit", "Eb/N0", "{:.1f} dB"),
        "wired": ("Per-symbol error rate on the line", "error rate", "BER {:.3f}"),
    }

    @staticmethod
    def sweep(
        x_values: Sequence[float],
        window_sizes: Sequence[int],
        message: bytes = DEFAULT_MESSAGE,
        channel_type: str = "wireless",
        mode: str = "BPSK",
        chunk_size: int = 8,
        prop_delay: int = 2,
        timeout_limit: int = 6,
        trials: int = 3,
        seed: Optional[int] = 20,
        progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Runs every (window size, channel quality) combination ``trials`` times and averages.

        ``x_values`` means Eb/N0 in dB on the wireless channel and the per-symbol error rate
        on the wired one. The two are not interchangeable: a baseband line code has no Eb/N0
        to sweep, so feeding decibels to B8ZS would silently benchmark an error-free link.

        Averaging matters here: a single run at one point is one draw from a random channel,
        so unaveraged curves are dominated by sampling noise rather than by the protocol
        behaviour they are meant to illustrate.
        """
        wired = channel_type == "wired"
        _, _, point_fmt = PerformanceAnalyzer.AXIS_LABELS[
            channel_type if channel_type in PerformanceAnalyzer.AXIS_LABELS else "wireless"
        ]

        results = {w: {"throughput": [], "goodput": [], "delivery": []} for w in window_sizes}
        run_id = 0

        for w in window_sizes:
            for x in x_values:
                throughput_runs, goodput_runs, delivery_runs = [], [], []
                for t in range(trials):
                    run_id += 1
                    sim = SelectiveRepeatSimulation(
                        encrypted_message=message,
                        channel_type=channel_type,
                        mode=mode,
                        snr_db=None if wired else float(x),
                        error_rate=float(x) if wired else None,
                        window_size=int(w),
                        timeout_limit=timeout_limit,
                        chunk_size=chunk_size,
                        prop_delay=prop_delay,
                        verbose=False,
                        seed=None if seed is None else seed + run_id,
                    )
                    _, throughput, goodput = sim.run(max_ticks=4000)
                    throughput_runs.append(throughput)
                    goodput_runs.append(goodput)
                    delivery_runs.append(sim.stats.frames_delivered / max(sim.num_frames, 1))

                results[w]["throughput"].append(float(np.mean(throughput_runs)))
                results[w]["goodput"].append(float(np.mean(goodput_runs)))
                results[w]["delivery"].append(float(np.mean(delivery_runs)))

                if progress:
                    progress(f"Window {w} @ {point_fmt.format(x)} -> "
                             f"goodput {np.mean(goodput_runs):.1f} bits/tick")

        return results

    @staticmethod
    def build_figure(
        x_values: Sequence[float],
        window_sizes: Sequence[int],
        results: dict,
        channel_type: str = "wireless",
        mode: str = "BPSK",
        figsize: Tuple[float, float] = (13.0, 5.2),
    ):
        """Renders the two comparative charts required by section 5.1."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        x_label, axis_name, _ = PerformanceAnalyzer.AXIS_LABELS[
            channel_type if channel_type in PerformanceAnalyzer.AXIS_LABELS else "wireless"
        ]

        for w in window_sizes:
            axes[0].plot(x_values, results[w]["throughput"], marker="o", linewidth=2, label=f"W = {w}")
            axes[1].plot(x_values, results[w]["goodput"], marker="s", linewidth=2, label=f"W = {w}")

        axes[0].set_title(f"Throughput vs {axis_name} - {mode} over {channel_type}",
                          fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Throughput (all channel bits / tick)", fontsize=10)
        axes[1].set_title(f"Goodput vs {axis_name} - {mode} over {channel_type}",
                          fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Goodput (delivered payload bits / tick)", fontsize=10)

        for ax in axes:
            ax.set_xlabel(x_label, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(title="Sliding window")

        fig.tight_layout()
        return fig

    @staticmethod
    def run_analysis(
        output_filepath: str = "performance_analysis_throughput_goodput.png",
        x_values: Optional[Iterable[float]] = None,
        window_sizes: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Runs the default sweep and writes the chart to disk."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib is required for performance analysis.")
            print("Install it with: python -m pip install matplotlib")
            return None

        channel_type = kwargs.get("channel_type", "wireless")
        if x_values is not None:
            x_list: List[float] = list(x_values)
        elif channel_type == "wired":
            x_list = [round(v, 3) for v in np.arange(0.0, 0.061, 0.01)]
        else:
            # The interesting region for BPSK is the waterfall below ~10 dB; above that the
            # channel is effectively error free and every curve flattens out.
            x_list = list(np.arange(0.0, 12.1, 1.5))
        windows = list(window_sizes) if window_sizes is not None else [1, 2, 4, 8]

        print("Starting batch performance simulations for analysis...")
        results = PerformanceAnalyzer.sweep(x_list, windows, progress=print, **kwargs)
        fig = PerformanceAnalyzer.build_figure(x_list, windows, results,
                                               channel_type,
                                               kwargs.get("mode", "BPSK"))
        fig.savefig(output_filepath, dpi=150)
        plt.close(fig)
        print(f"Performance plots successfully generated and saved to '{output_filepath}'")
        return results
