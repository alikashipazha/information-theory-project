"""
Physical Layer Module (Specification sections 2, 3.1 and 3.2).

Wired media use bipolar line codes (B8ZS, HDB3) whose job is to keep the receiver
synchronised and the average DC level at zero. Wireless media use carrier modulation
(BPSK, 16-QAM) over an AWGN channel.

SNR convention: every SNR figure in this project is Eb/N0, energy per *information bit*
over noise power density. This matters because 16-QAM carries four bits per symbol, so
quoting it in Es/N0 instead would flatter it by a constant 10*log10(4) = 6.02 dB for no
physical reason and would make the two modulations impossible to compare on one axis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MODES_BY_CHANNEL: Dict[str, Tuple[str, ...]] = {
    "wired": ("B8ZS", "HDB3"),
    "wireless": ("BPSK", "16-QAM"),
}

BITS_PER_SYMBOL: Dict[str, int] = {"BPSK": 1, "16-QAM": 4}

# Mean of I^2 + Q^2 over the 16 equally likely points of the {+-1, +-3}^2 constellation.
QAM16_MEAN_SYMBOL_ENERGY = 10.0

# 16-QAM slicer thresholds: the midpoints between the four amplitude levels.
QAM16_DECISION_LEVELS: Tuple[float, ...] = (-2.0, 0.0, 2.0)


@dataclass
class ChannelResult:
    """
    What the receiver recovered, plus the signals the GUI needs for section 3.1/3.2.

    ``tx_quadrature`` is only populated by modulations that actually use a quadrature
    component, so a caller can tell a genuine two-dimensional constellation from a
    baseband line code without inspecting the mode string.
    """

    rx_bits: List[int]
    tx_waveform: List[float] = field(default_factory=list)
    rx_waveform: List[float] = field(default_factory=list)
    tx_quadrature: Optional[List[float]] = None
    rx_quadrature: Optional[List[float]] = None

    @property
    def is_quadrature(self) -> bool:
        return self.tx_quadrature is not None


class PhysicalLayerWired:
    """
    Line coding schemes for wired communication.
    Solves DC component accumulation and synchronization loss.
    """

    @staticmethod
    def b8zs_encode(bits: List[int]) -> List[int]:
        """
        Bipolar with 8-Zero Substitution (North American standard).
        Replaces 8 consecutive zeros with the 000VB0VB pattern, which deliberately
        violates the alternating-polarity rule so the decoder can recognise it.
        """
        encoded = []
        last_polarity = -1
        i = 0
        while i < len(bits):
            if i <= len(bits) - 8 and bits[i:i + 8] == [0] * 8:
                p = last_polarity
                # V has the same polarity as the previous pulse (the violation), B alternates.
                # The pattern ends on polarity p, so last_polarity is already correct.
                encoded.extend([0, 0, 0, p, -p, 0, -p, p])
                i += 8
            else:
                if bits[i] == 1:
                    last_polarity = -last_polarity
                    encoded.append(last_polarity)
                else:
                    encoded.append(0)
                i += 1
        return encoded

    @staticmethod
    def b8zs_decode(sig: Sequence[float]) -> List[int]:
        """
        Decodes B8ZS line-coded signals back to binary.

        The substitution is recognised by matching the 000VB0VB signature directly rather
        than by tracking the polarity of the previous pulse. Tracking is what used to break
        on a stream that opens with eight zeros: at that point no pulse has been seen yet,
        so the violation is invisible and the pattern decodes as four spurious ones.

        Matching is unambiguous because the signature contains two same-polarity pulses
        separated by a single zero, which alternate-mark inversion can never produce from
        real data.
        """
        symbols = list(sig)
        bits: List[int] = []
        i = 0
        total = len(symbols)
        while i < total:
            if i + 8 <= total:
                v = symbols[i + 3]
                if (symbols[i] == 0 and symbols[i + 1] == 0 and symbols[i + 2] == 0
                        and v != 0
                        and symbols[i + 4] == -v
                        and symbols[i + 5] == 0
                        and symbols[i + 6] == -v
                        and symbols[i + 7] == v):
                    bits.extend([0] * 8)
                    i += 8
                    continue
            bits.append(1 if symbols[i] != 0 else 0)
            i += 1
        return bits

    @staticmethod
    def hdb3_encode(bits: List[int]) -> List[int]:
        """
        High-Density Bipolar 3-zero (European standard).
        Replaces 4 consecutive zeros with 000V or B00V, alternating between the two so
        that the substitutions themselves do not introduce a DC offset.
        """
        encoded = []
        last_polarity = -1
        pulses_since_last_v = 0
        i = 0
        while i < len(bits):
            if i <= len(bits) - 4 and bits[i:i + 4] == [0, 0, 0, 0]:
                if pulses_since_last_v % 2 == 0:
                    b_pol = -last_polarity
                    encoded.extend([b_pol, 0, 0, b_pol])
                    last_polarity = b_pol
                else:
                    encoded.extend([0, 0, 0, last_polarity])
                pulses_since_last_v = 0
                i += 4
            else:
                if bits[i] == 1:
                    last_polarity = -last_polarity
                    encoded.append(last_polarity)
                    pulses_since_last_v += 1
                else:
                    encoded.append(0)
                i += 1
        return encoded

    @staticmethod
    def hdb3_decode(sig: Sequence[float]) -> List[int]:
        """Decodes HDB3 line-coded signals back to binary."""
        decoded = list(sig)
        last_non_zero_val = 0
        i = 0
        while i < len(decoded):
            if decoded[i] != 0:
                if last_non_zero_val != 0 and decoded[i] == last_non_zero_val:
                    if i >= 3 and decoded[i - 3] != 0 and decoded[i - 2] == 0 and decoded[i - 1] == 0:
                        decoded[i - 3] = 0
                    decoded[i] = 0
                last_non_zero_val = decoded[i]
            i += 1
        return [1 if x != 0 else 0 for x in decoded]

    @staticmethod
    def dc_imbalance(waveform: Sequence[float]) -> int:
        """Running sum of pulse polarities; a good line code keeps this near zero."""
        return int(sum(1 if s > 0 else -1 if s < 0 else 0 for s in waveform))

    @staticmethod
    def longest_zero_run(waveform: Sequence[float]) -> int:
        """
        Longest stretch with no pulse, i.e. no timing information for the receiver.
        This is the quantity B8ZS and HDB3 exist to bound.
        """
        longest = current = 0
        for symbol in waveform:
            current = 0 if symbol != 0 else current + 1
            longest = max(longest, current)
        return longest


class PhysicalLayerWireless:
    """
    Modulation schemes for wireless communication.
    Maps digital bits to complex constellations and vice versa.
    """

    gray_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    inv_gray_map = {v: k for k, v in gray_map.items()}

    @staticmethod
    def bpsk_modulate(bits: List[int]) -> List[float]:
        """Maps 1 -> 1.0 and 0 -> -1.0."""
        return [1.0 if b == 1 else -1.0 for b in bits]

    @staticmethod
    def bpsk_demodulate(noisy: Sequence[float]) -> List[int]:
        """Performs threshold detection at 0."""
        return [1 if x > 0 else 0 for x in noisy]

    @classmethod
    def constellation_points(cls) -> List[complex]:
        """The 16 ideal symbol locations, for drawing the reference grid."""
        levels = sorted(cls.gray_map.values())
        return [complex(i, q) for i in levels for q in levels]

    @classmethod
    def qam16_modulate(cls, bits: List[int]) -> Tuple[List[complex], int]:
        """
        Maps groups of 4 bits to 16-QAM complex symbols using Gray coding.
        Returns the symbols plus the number of padding bits that were appended, which the
        demodulator needs in order to strip exactly the right amount again.
        """
        pad_len = (-len(bits)) % 4
        padded_bits = list(bits) + [0] * pad_len

        symbols = []
        for i in range(0, len(padded_bits), 4):
            in_phase = cls.gray_map[(padded_bits[i], padded_bits[i + 1])]
            quadrature = cls.gray_map[(padded_bits[i + 2], padded_bits[i + 3])]
            symbols.append(complex(in_phase, quadrature))
        return symbols, pad_len

    @staticmethod
    def _quantize(value: float) -> int:
        if value < -2:
            return -3
        if value < 0:
            return -1
        if value < 2:
            return 1
        return 3

    @classmethod
    def qam16_demodulate(cls, noisy_symbols: Sequence[complex], pad_len: int) -> List[int]:
        """
        Demodulates 16-QAM symbols back to binary bits using minimum-distance slicing.
        ``pad_len`` is the number of bits the modulator appended, not the leftover count.
        """
        bits: List[int] = []
        for symbol in noisy_symbols:
            b_i = cls.inv_gray_map[cls._quantize(symbol.real)]
            b_q = cls.inv_gray_map[cls._quantize(symbol.imag)]
            bits.extend(list(b_i) + list(b_q))

        if pad_len:
            bits = bits[:len(bits) - pad_len]
        return bits


class CommunicationChannel:
    """Simulates transmission over wired or wireless media."""

    WIRED_ERROR_RATE = 0.0005

    @staticmethod
    def validate_link(channel_type: str, mode: str, snr_db: Optional[float] = None) -> None:
        """
        Rejects impossible configurations up front.

        Without this a typo such as wireless/B8ZS used to fall through to an ideal
        error-free channel and quietly produce perfect results, and a missing SNR raised a
        TypeError from inside the noise generator instead of naming the real problem.
        """
        if channel_type not in MODES_BY_CHANNEL:
            raise ValueError(
                f"Unknown channel type {channel_type!r}; expected one of "
                f"{sorted(MODES_BY_CHANNEL)}."
            )
        allowed = MODES_BY_CHANNEL[channel_type]
        if mode not in allowed:
            raise ValueError(
                f"Mode {mode!r} is not available on the {channel_type} channel; "
                f"expected one of {list(allowed)}."
            )
        if channel_type == "wireless" and snr_db is None:
            raise ValueError("The wireless channel needs an Eb/N0 value in dB (snr_db).")

    @staticmethod
    def noise_std_per_component(mode: str, snr_db: float) -> float:
        """
        Converts an Eb/N0 figure in dB to the standard deviation of the AWGN added to each
        signal component.

        Both modulations are quoted in Eb/N0 so their curves share one axis honestly. For
        BPSK a symbol carries one bit, so Eb = Es = 1. For 16-QAM a symbol carries four,
        so Eb = 10/4 = 2.5.
        """
        ebn0_linear = 10 ** (snr_db / 10.0)
        energy_per_symbol = 1.0 if mode == "BPSK" else QAM16_MEAN_SYMBOL_ENERGY
        energy_per_bit = energy_per_symbol / BITS_PER_SYMBOL[mode]
        n0 = energy_per_bit / ebn0_linear
        return float(np.sqrt(n0 / 2.0))

    @staticmethod
    def transmit(
        bits: List[int],
        channel_type: str,
        mode: str,
        snr_db: Optional[float] = None,
        return_waveform: bool = False,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        error_rate: Optional[float] = None,
    ) -> Union[List[int], ChannelResult]:
        """
        Sends a bit stream across the selected medium and returns the recovered bits, or a
        full :class:`ChannelResult` when ``return_waveform`` is set.

        Callers that need statistically independent noise across repeated calls (such as
        ARQ retransmissions) must pass a single long-lived ``rng``; passing ``seed``
        instead restarts the same stream on every call and reproduces identical noise.

        ``error_rate`` is the per-symbol corruption probability of the wired channel and is
        ignored by the wireless path, which derives its noise from ``snr_db`` instead.
        """
        CommunicationChannel.validate_link(channel_type, mode, snr_db)

        if rng is None:
            rng = np.random.default_rng(seed)

        if channel_type == "wired":
            encode, decode = (
                (PhysicalLayerWired.b8zs_encode, PhysicalLayerWired.b8zs_decode)
                if mode == "B8ZS"
                else (PhysicalLayerWired.hdb3_encode, PhysicalLayerWired.hdb3_decode)
            )
            rate = CommunicationChannel.WIRED_ERROR_RATE if error_rate is None else error_rate
            clean = [float(s) for s in encode(bits)]
            noisy = CommunicationChannel._apply_wired_noise(clean, rng, rate)
            result = ChannelResult(rx_bits=decode(noisy), tx_waveform=clean, rx_waveform=noisy)

        elif mode == "BPSK":
            noise_std = CommunicationChannel.noise_std_per_component(mode, snr_db)
            clean = PhysicalLayerWireless.bpsk_modulate(bits)
            noise = rng.normal(0.0, noise_std, len(clean))
            noisy = [s + n for s, n in zip(clean, noise)]
            result = ChannelResult(
                rx_bits=PhysicalLayerWireless.bpsk_demodulate(noisy),
                tx_waveform=clean,
                rx_waveform=noisy,
            )

        else:
            noise_std = CommunicationChannel.noise_std_per_component(mode, snr_db)
            symbols, pad_len = PhysicalLayerWireless.qam16_modulate(bits)
            noise_i = rng.normal(0.0, noise_std, len(symbols))
            noise_q = rng.normal(0.0, noise_std, len(symbols))
            noisy_symbols = [s + complex(ni, nq) for s, ni, nq in zip(symbols, noise_i, noise_q)]
            result = ChannelResult(
                rx_bits=PhysicalLayerWireless.qam16_demodulate(noisy_symbols, pad_len),
                tx_waveform=[s.real for s in symbols],
                rx_waveform=[s.real for s in noisy_symbols],
                tx_quadrature=[s.imag for s in symbols],
                rx_quadrature=[s.imag for s in noisy_symbols],
            )

        return result if return_waveform else result.rx_bits

    @staticmethod
    def _apply_wired_noise(waveform: Sequence[float], rng: np.random.Generator,
                           error_rate: float) -> List[float]:
        """
        Corrupts line-coded symbols at the given rate.

        A hit moves the symbol to one of the two *other* levels, so both a lost pulse and a
        spurious pulse are possible. The previous model simply negated the symbol, which left
        every zero untouched because -0 == 0 — on a bipolar line where most symbols are zero
        that made the majority of the waveform immune to noise and the wired channel far
        quieter than it should have been.
        """
        levels = (-1.0, 0.0, 1.0)
        noisy: List[float] = []
        for symbol in waveform:
            if rng.random() >= error_rate:
                noisy.append(symbol)
                continue
            alternatives = [level for level in levels if level != symbol]
            noisy.append(alternatives[int(rng.integers(len(alternatives)))])
        return noisy
