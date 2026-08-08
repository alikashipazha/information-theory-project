import importlib
import inspect

import numpy as np
import pytest

import src.performance_analysis as pa

from src.data_link_layer import DataLinkUtils, Frame, FrameDelimiter
from src.security import SecurityLayer
from src.physical_layer import (
    QAM16_MEAN_SYMBOL_ENERGY,
    ChannelResult,
    CommunicationChannel,
    PhysicalLayerWired,
    PhysicalLayerWireless,
)
from src.arq_protocol import SelectiveRepeatSimulation


def test_repo_root_launcher_module_imports():
    run_module = importlib.import_module("run")
    assert callable(getattr(run_module, "main", None))


def test_security_round_trip():
    payload = b"hello simulator"
    key = "secure-key"
    encrypted = SecurityLayer.encrypt_xor_rotational(payload, key)
    decrypted = SecurityLayer.decrypt_xor_rotational(encrypted, key)
    assert decrypted == payload


def test_frame_round_trip_and_crc_validation():
    frame = Frame(seq_num=7, frame_type="DATA", payload=b"abc")
    bits = frame.to_bits()
    decoded = Frame.from_bits(bits)
    assert decoded is not None
    assert decoded.seq_num == frame.seq_num
    assert decoded.frame_type == frame.frame_type
    assert decoded.payload == frame.payload


def test_hamming_corrects_a_single_bit_error():
    bits = Frame(seq_num=5, frame_type="DATA", payload=b"payload").to_bits()
    bits[8 + 12 * 6 + 4] ^= 1  # one bit inside the 7th Hamming block of the body
    result = Frame.decode(bits)
    assert result.payload_valid
    assert result.corrected_bits == 1
    assert result.payload == b"payload"


def test_hamming_and_crc_reject_corrupted_frame():
    frame = Frame(seq_num=2, frame_type="DATA", payload=b"xyz")
    bits = frame.to_bits()
    bits[len(bits) // 2] ^= 1
    bits[len(bits) // 2 + 3] ^= 1
    decoded = Frame.from_bits(bits)
    assert decoded is None or decoded.payload == b"xyz"


def test_damaged_payload_keeps_header_identifiable_for_nak():
    """A NAK can only name a frame if the header survives independently of the payload."""
    bits = Frame(seq_num=9, frame_type="DATA", payload=b"abcd").to_bits()
    block = 8 + 12 * 9
    bits[block + 2] ^= 1
    bits[block + 5] ^= 1  # two errors in one block: FEC miscorrects, CRC-32 must catch it
    result = Frame.decode(bits)
    assert result.header_valid
    assert not result.payload_valid
    assert result.seq_num == 9
    assert result.error == "PAYLOAD_CRC"


def test_damaged_header_is_not_trusted():
    bits = Frame(seq_num=9, frame_type="DATA", payload=b"abcd").to_bits()
    bits[8 + 1] ^= 1
    bits[8 + 4] ^= 1
    result = Frame.decode(bits)
    assert not result.header_valid
    assert result.seq_num is None


def test_retransmissions_see_independent_noise():
    """A shared generator must not replay the same noise pattern on every attempt."""
    rng = np.random.default_rng(7)
    tx = Frame(0, "DATA", b"abcd").to_bits()
    outputs = [CommunicationChannel.transmit(tx, "wireless", "BPSK", 1.0, rng=rng) for _ in range(5)]
    assert any(out != outputs[0] for out in outputs[1:])


def test_channel_is_reproducible_for_a_given_seed():
    tx = Frame(0, "DATA", b"abcd").to_bits()
    first = CommunicationChannel.transmit(tx, "wireless", "BPSK", 1.0, rng=np.random.default_rng(3))
    second = CommunicationChannel.transmit(tx, "wireless", "BPSK", 1.0, rng=np.random.default_rng(3))
    assert first == second


def test_hdb3_round_trip_over_random_streams():
    rng = np.random.default_rng(11)
    for _ in range(200):
        bits = [int(b) for b in rng.integers(0, 2, size=int(rng.integers(1, 60)))]
        assert PhysicalLayerWired.hdb3_decode(PhysicalLayerWired.hdb3_encode(bits)) == bits


def test_delimiter_finds_frames_inside_a_continuous_stream():
    """
    Section 4.1 asks the receiver to locate frames itself. The stream is padded with junk and
    idle flags so the delimiter has to resynchronise rather than rely on tidy inputs.
    """
    payloads = [b"one", b"two", b"three"]
    wire = [1, 0, 1, 1, 0]  # leading junk before any flag
    for index, payload in enumerate(payloads):
        wire += Frame(index, "DATA", payload).to_bits()

    delimiter = FrameDelimiter()
    frames = delimiter.feed(wire)

    decoded = [Frame.decode(bits) for bits in frames]
    assert [d.payload for d in decoded] == payloads
    assert all(d.header_valid and d.payload_valid for d in decoded)
    assert delimiter.frames_found == 3
    assert delimiter.discarded_bits == 5  # exactly the junk, nothing more


def test_delimiter_waits_for_a_frame_split_across_reads():
    """A frame arriving in two pieces must not be dropped or half-parsed."""
    bits = Frame(4, "DATA", b"split").to_bits()
    cut = len(bits) // 2
    delimiter = FrameDelimiter()

    assert delimiter.feed(bits[:cut]) == []
    frames = delimiter.feed(bits[cut:])

    assert len(frames) == 1
    assert Frame.decode(frames[0]).payload == b"split"


def test_delimiter_recovers_the_next_frame_after_a_damaged_flag():
    """Losing a flag should cost the damaged frame, not every frame that follows it."""
    good = Frame(1, "DATA", b"intact")
    damaged = Frame(0, "DATA", b"broken").to_bits()
    damaged[2] ^= 1  # break the opening flag so this frame cannot be delimited

    delimiter = FrameDelimiter()
    frames = delimiter.feed(damaged + good.to_bits())

    payloads = [Frame.decode(bits).payload for bits in frames]
    assert b"intact" in payloads
    assert delimiter.resyncs >= 1


def test_bit_stuffing_never_emits_six_consecutive_ones():
    """This is the property that lets a flag search be unambiguous."""
    stuffed = DataLinkUtils.bit_stuff([1] * 40)
    assert not any(stuffed[i:i + 6] == [1] * 6 for i in range(len(stuffed) - 5))


def test_unstuffing_keeps_alignment_when_a_stuff_bit_is_corrupted():
    """
    The old unstuffer only removed the following bit when it was still 0, so a single flipped
    stuff bit shifted the rest of the frame and destroyed it. Length must now be unaffected.
    """
    original = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    stuffed, positions = DataLinkUtils.bit_stuff_ex(original)
    assert positions, "this fixture must actually contain stuffing"

    damaged = list(stuffed)
    for position in positions:
        damaged[position] ^= 1  # flip every inserted 0 into a 1

    recovered = DataLinkUtils.bit_unstuff(damaged)
    assert len(recovered) == len(original)
    assert DataLinkUtils.bit_unstuff(stuffed) == original


def test_wired_errors_can_also_create_and_erase_pulses():
    """
    The old model negated the symbol, and since -0 == 0 every zero was immune. On a bipolar
    line most symbols are zero, so most of the waveform could not be corrupted at all.
    """
    clean = [0.0] * 400
    noisy = CommunicationChannel._apply_wired_noise(clean, np.random.default_rng(4), 0.5)
    assert any(symbol != 0.0 for symbol in noisy), "zeros must be corruptible"

    pulses = [1.0] * 400
    hit = CommunicationChannel._apply_wired_noise(pulses, np.random.default_rng(4), 0.5)
    assert any(symbol == 0.0 for symbol in hit), "a pulse must be able to vanish"
    assert all(symbol in (-1.0, 0.0, 1.0) for symbol in noisy + hit)


def test_wired_error_rate_is_configurable_end_to_end():
    bits = Frame(0, "DATA", b"wired path").to_bits()
    quiet = CommunicationChannel.transmit(bits, "wired", "B8ZS", rng=np.random.default_rng(8),
                                          error_rate=0.0)
    assert quiet == bits

    loud = CommunicationChannel.transmit(bits, "wired", "HDB3", rng=np.random.default_rng(8),
                                         error_rate=0.4)
    assert loud != bits


def test_b8zs_round_trip_over_random_streams():
    """Zero-heavy streams are the whole point of B8ZS, so bias the generator towards zeros."""
    rng = np.random.default_rng(19)
    for _ in range(400):
        length = int(rng.integers(1, 70))
        bits = [int(rng.random() < 0.25) for _ in range(length)]
        encoded = PhysicalLayerWired.b8zs_encode(bits)
        assert PhysicalLayerWired.b8zs_decode(encoded) == bits, bits


def test_b8zs_decodes_a_substitution_at_the_very_start():
    """
    The old decoder tracked the previous pulse polarity to spot the violation, so a stream
    opening with eight zeros decoded as four spurious ones.
    """
    bits = [0] * 8 + [1, 0, 1]
    encoded = PhysicalLayerWired.b8zs_encode(bits)
    assert PhysicalLayerWired.b8zs_decode(encoded) == bits


def test_b8zs_bounds_the_run_of_zeros_it_emits():
    """The synchronisation guarantee: no line code output may go 8 symbols without a pulse."""
    encoded = PhysicalLayerWired.b8zs_encode([0] * 40)
    assert PhysicalLayerWired.longest_zero_run(encoded) < 8


def test_both_modulations_are_quoted_in_the_same_eb_n0_units():
    """
    16-QAM used to be quoted in Es/N0 while BPSK was in Eb/N0, a silent 6.02 dB handicap
    that made the two curves incomparable. Both must now derive N0 from energy per bit.
    """
    for snr_db in (0.0, 6.0, 12.0):
        ebn0 = 10 ** (snr_db / 10.0)
        bpsk = CommunicationChannel.noise_std_per_component("BPSK", snr_db)
        qam = CommunicationChannel.noise_std_per_component("16-QAM", snr_db)
        assert bpsk == pytest.approx(np.sqrt(1.0 / (2.0 * ebn0)))
        assert qam == pytest.approx(np.sqrt((QAM16_MEAN_SYMBOL_ENERGY / 4.0) / (2.0 * ebn0)))


def test_16qam_is_noisier_than_bpsk_at_equal_eb_n0():
    """Denser constellations must cost accuracy; equal Eb/N0 is the only fair comparison."""
    bits = [int(b) for b in np.random.default_rng(5).integers(0, 2, size=4000)]

    def error_rate(mode, snr_db):
        rx = CommunicationChannel.transmit(bits, "wireless", mode, snr_db,
                                           rng=np.random.default_rng(77))
        return sum(a != b for a, b in zip(bits, rx)) / len(bits)

    assert error_rate("16-QAM", 6.0) > error_rate("BPSK", 6.0)
    # Both must still improve as the channel gets cleaner.
    assert error_rate("16-QAM", 16.0) < error_rate("16-QAM", 6.0)


def test_invalid_channel_and_mode_combinations_are_rejected():
    """These used to fall through to an error-free channel and report a flawless run."""
    bits = [1, 0, 1, 1]
    with pytest.raises(ValueError, match="not available on the wireless channel"):
        CommunicationChannel.transmit(bits, "wireless", "B8ZS", 8.0)
    with pytest.raises(ValueError, match="not available on the wired channel"):
        CommunicationChannel.transmit(bits, "wired", "BPSK")
    with pytest.raises(ValueError, match="Unknown channel type"):
        CommunicationChannel.transmit(bits, "fibre", "BPSK", 8.0)
    with pytest.raises(ValueError, match="needs an Eb/N0"):
        CommunicationChannel.transmit(bits, "wireless", "BPSK", None)
    with pytest.raises(ValueError):
        SelectiveRepeatSimulation(b"abc", channel_type="wireless", mode="HDB3", verbose=False)


def test_16qam_exposes_the_quadrature_component_for_the_constellation():
    """Plotting only the in-phase axis hides half of what a 16-QAM symbol carries."""
    bits = [int(b) for b in np.random.default_rng(2).integers(0, 2, size=64)]

    qam = CommunicationChannel.transmit(bits, "wireless", "16-QAM", 20.0,
                                        return_waveform=True, rng=np.random.default_rng(1))
    assert isinstance(qam, ChannelResult)
    assert qam.is_quadrature
    assert len(qam.tx_quadrature) == len(qam.tx_waveform) == len(bits) // 4
    assert {abs(v) for v in qam.tx_quadrature} <= {1.0, 3.0}

    bpsk = CommunicationChannel.transmit(bits, "wireless", "BPSK", 20.0,
                                         return_waveform=True, rng=np.random.default_rng(1))
    assert not bpsk.is_quadrature

    assert len(PhysicalLayerWireless.constellation_points()) == 16


def test_clean_channel_delivers_everything():
    payload = b"selective repeat over a quiet channel"
    sim = SelectiveRepeatSimulation(payload, "wireless", "BPSK", snr_db=25.0,
                                    window_size=4, timeout_limit=6, prop_delay=2,
                                    verbose=False, seed=1)
    reconstructed, throughput, goodput = sim.run(max_ticks=800)
    assert sim.success
    assert reconstructed == payload
    assert goodput > 0
    assert goodput <= throughput


def test_ack_and_nak_frames_traverse_the_channel():
    """Control frames must be transmitted, otherwise they can never be lost or corrupted."""
    calls = []
    original = CommunicationChannel.__dict__["transmit"].__func__

    def spy(bits, *args, **kwargs):
        calls.append(len(bits))
        return original(bits, *args, **kwargs)

    CommunicationChannel.transmit = staticmethod(spy)
    try:
        sim = SelectiveRepeatSimulation(b"abcdefgh", "wireless", "BPSK", 25.0, 2, 6,
                                        prop_delay=2, verbose=False, seed=1)
        sim.run(max_ticks=200)
    finally:
        CommunicationChannel.transmit = staticmethod(original)

    stats = sim.stats
    assert stats.acks_sent > 0
    assert len(calls) == stats.data_transmissions + stats.acks_sent + stats.naks_sent


def test_timer_recovers_a_lost_acknowledgement():
    sim = SelectiveRepeatSimulation(b"abcdefgh", "wireless", "BPSK", 25.0, 1, 3,
                                    verbose=False, seed=2, drop_ack_for_sequences={0})
    reconstructed, _, _ = sim.run(max_ticks=120)
    assert reconstructed == b"abcdefgh"
    assert sim.stats.timeouts >= 1
    assert sim.stats.retransmissions >= 1


def test_goodput_counts_only_delivered_payload():
    """A run that loses frames must report a lower goodput than a clean run."""
    clean = SelectiveRepeatSimulation(b"x" * 48, "wireless", "BPSK", 25.0, 4, 6,
                                      prop_delay=2, verbose=False, seed=4)
    clean.run(max_ticks=800)

    noisy = SelectiveRepeatSimulation(b"x" * 48, "wireless", "BPSK", 0.0, 4, 6,
                                      prop_delay=2, verbose=False, seed=4)
    noisy.run(max_ticks=800)

    assert clean.success
    assert clean.stats.delivered_payload_bits == len(b"x" * 48) * 8
    assert noisy.stats.delivered_payload_bits == noisy.stats.frames_delivered * noisy.chunk_size * 8
    assert noisy.stats.goodput < clean.stats.goodput


def test_undeliverable_payload_is_reported_not_silently_truncated():
    sim = SelectiveRepeatSimulation(b"ABCDEFGHIJKLMNOPQRSTUVWX", "wireless", "BPSK",
                                    snr_db=-6.0, window_size=3, timeout_limit=3,
                                    max_retries=2, verbose=False, seed=7)
    reconstructed, _, _ = sim.run(max_ticks=400)
    assert not sim.success
    assert len(reconstructed) < 24
    assert sim.failed_frames
    assert any("LOST" in line for line in sim.history)


def test_simulation_terminates_instead_of_spinning_to_the_tick_budget():
    max_ticks = 500
    sim = SelectiveRepeatSimulation(b"ABCDEFGH" * 3, "wireless", "BPSK", snr_db=-6.0,
                                    window_size=2, timeout_limit=3, max_retries=2,
                                    verbose=False, seed=9)
    sim.run(max_ticks=max_ticks)
    assert sim.stats.ticks < max_ticks


def test_larger_window_pipelines_the_link():
    """Utilisation must improve with the window until it covers the round trip."""
    payload = b"y" * 160
    ticks = {}
    for window in (1, 2, 5):
        sim = SelectiveRepeatSimulation(payload, "wireless", "BPSK", 25.0, window, 8,
                                        chunk_size=8, prop_delay=2, verbose=False, seed=5)
        sim.run(max_ticks=4000)
        assert sim.success
        ticks[window] = sim.stats.ticks
    assert ticks[1] > ticks[2] > ticks[5]


def test_modular_sequence_numbers_survive_more_than_one_wrap():
    payload = bytes((i * 7 + 3) & 0xFF for i in range(1400))  # 350 frames at chunk_size 4
    sim = SelectiveRepeatSimulation(payload, "wireless", "BPSK", 25.0, 4, 6,
                                    chunk_size=4, verbose=False, seed=4)
    reconstructed, _, _ = sim.run(max_ticks=8000)
    assert sim.num_frames > 256
    assert sim.success
    assert reconstructed == payload


def test_end_to_end_encrypted_transmission_over_every_mode():
    plaintext = "End-to-end secure transmission across every configured medium."
    key = "SecureXORKey"
    ciphertext = SecurityLayer.encrypt_xor_rotational(plaintext.encode("utf-8"), key)

    for channel, mode, snr in [("wireless", "BPSK", 25.0), ("wireless", "16-QAM", 30.0),
                               ("wired", "B8ZS", None), ("wired", "HDB3", None)]:
        sim = SelectiveRepeatSimulation(ciphertext, channel, mode, snr, 4, 6,
                                        chunk_size=8, prop_delay=2, verbose=False, seed=6)
        received, _, _ = sim.run(max_ticks=2000)
        recovered = SecurityLayer.decrypt_xor_rotational(received, key)
        assert sim.success, f"{channel}/{mode} failed to deliver every frame"
        assert recovered.decode("utf-8") == plaintext


def test_bit_stuffing_never_emits_a_flag_pattern():
    rng = np.random.default_rng(2)
    for _ in range(50):
        bits = [int(b) for b in rng.integers(0, 2, size=200)]
        stuffed = DataLinkUtils.bit_stuff(bits)
        assert DataLinkUtils.bit_unstuff(stuffed) == bits
        run = 0
        for bit in stuffed:
            run = run + 1 if bit == 1 else 0
            assert run < 6


def test_retry_budget_is_configurable_and_actually_bounds_the_attempts():
    """
    The retry budget used to be a literal buried in the sender, so the operator could not
    trade delivery against latency. Each configured budget must cap the attempts per frame.
    """
    attempts = {}
    for budget in (2, 5):
        sim = SelectiveRepeatSimulation(b"ABCDEFGHIJKL", "wireless", "BPSK", snr_db=-6.0,
                                        window_size=2, timeout_limit=3, max_retries=budget,
                                        verbose=False, seed=11)
        sim.run(max_ticks=1200)
        assert sim.max_retries == budget
        assert sim.failed_frames, "this fixture must be lossy enough to exhaust the budget"
        attempts[budget] = max(sim.retry_attempts.values())
        assert attempts[budget] <= budget

    assert attempts[5] > attempts[2], "a larger budget must buy more attempts"


def test_the_gui_exposes_the_retry_budget_as_a_configurable_control():
    """A knob nobody can reach is not configurable, so the GUI must own a widget for it."""
    gui = importlib.import_module("src.gui")
    source = inspect.getsource(gui.UnifiedSimulatorGUI)
    assert "self.retry_spin" in source
    assert "max_retries=retries" in source


def test_a_control_frame_with_a_repaired_header_is_not_acted_upon():
    """
    An ACK is nothing but a header. Hamming cannot tell a repair from a mis-repair, so a
    two-bit error makes it invent a plausible sequence number and acknowledge the wrong
    frame. Such a frame must be dropped and left to the timer, not believed.
    """
    ack_bits = Frame(3, "ACK", b"").to_bits()

    flag = len(Frame.FLAG)
    damaged = list(ack_bits)
    damaged[flag + 2] ^= 1  # inside the first Hamming block, which carries the sequence number

    result = Frame.decode(damaged)
    assert result.header_corrected, "the decoder must report where it applied a correction"

    sim = SelectiveRepeatSimulation(b"ABCDEFGH", "wireless", "BPSK", 25.0, 2, 6,
                                    verbose=False, seed=3)
    before = list(sim.frame_states)
    sim._sender_process(damaged, "ACK")

    assert sim.frame_states == before, "a repaired control header must not move the send window"
    assert sim.stats.acks_received == 0
    assert sim.stats.control_distrusted == 1
    assert any("CTRL-SUSPECT" in line for line in sim.history)


def test_an_untouched_control_frame_is_still_accepted():
    """The distrust rule must not throw away healthy acknowledgements."""
    sim = SelectiveRepeatSimulation(b"ABCDEFGH", "wireless", "BPSK", 25.0, 2, 6,
                                    verbose=False, seed=3)
    sim._transmit_data(0)
    sim._sender_process(Frame(0, "ACK", b"").to_bits(), "ACK")

    assert sim.stats.acks_received == 1
    assert sim.stats.control_distrusted == 0


def test_a_wired_sweep_varies_the_error_rate_instead_of_eb_n0():
    """
    The performance tab only ever swept Eb/N0, which is meaningless on a baseband line, so
    B8ZS and HDB3 were benchmarked over a channel that never made a mistake. A wired sweep
    must feed the x-axis into error_rate, leave snr_db unset, and degrade as it rises.
    """
    seen = []
    original = pa.SelectiveRepeatSimulation

    class Recorder(original):
        def __init__(self, **kwargs):
            seen.append((kwargs["snr_db"], kwargs["error_rate"]))
            super().__init__(**kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(pa, "SelectiveRepeatSimulation", Recorder)
    try:
        results = pa.PerformanceAnalyzer.sweep([0.0, 0.05], [2], channel_type="wired",
                                               mode="B8ZS", trials=1, seed=5)
    finally:
        monkey.undo()

    assert seen == [(None, 0.0), (None, 0.05)]
    assert results[2]["goodput"][0] > results[2]["goodput"][1]


def test_a_wireless_sweep_still_varies_eb_n0():
    """The wired branch must not hijack the axis the wireless channel actually needs."""
    seen = []
    original = pa.SelectiveRepeatSimulation

    class Recorder(original):
        def __init__(self, **kwargs):
            seen.append((kwargs["snr_db"], kwargs["error_rate"]))
            super().__init__(**kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(pa, "SelectiveRepeatSimulation", Recorder)
    try:
        pa.PerformanceAnalyzer.sweep([2.0, 9.0], [1], trials=1, seed=5)
    finally:
        monkey.undo()

    assert seen == [(2.0, None), (9.0, None)]


@pytest.mark.skipif(not pa.HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_the_chart_names_the_axis_the_channel_was_actually_swept_on():
    """A wired chart labelled Eb/N0 would misreport what the numbers underneath it mean."""
    x = [0.0, 0.05]
    results = {1: {"throughput": [10.0, 2.0], "goodput": [8.0, 1.0], "delivery": [1.0, 0.2]}}

    wired = pa.PerformanceAnalyzer.build_figure(x, [1], results, "wired", "B8ZS")
    assert "error rate" in wired.axes[0].get_xlabel()
    assert "Eb/N0" not in wired.axes[0].get_title()

    wireless = pa.PerformanceAnalyzer.build_figure(x, [1], results, "wireless", "BPSK")
    assert "Eb/N0" in wireless.axes[0].get_xlabel()

    pa.plt.close(wired)
    pa.plt.close(wireless)
