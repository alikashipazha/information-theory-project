# End-to-End Secure Communications Network Simulator

A modular Python simulator that carries a message through a **Security layer**, a **Data Link layer** and a **Physical layer** over a deliberately impaired channel, and shows every step of that journey live in a GUI.

The project implements the specification in `docs/Project_CN1.pdf`. Section numbers used in the code comments, the GUI labels and this file (for example `§4.1`) refer to that document, so any requirement can be traced to the code that satisfies it and to the panel that demonstrates it.

---

## 🚀 Implemented capabilities

### 🔒 1. Security layer (§5.3)

* **Rotational XOR cipher with dynamic keying.** The payload is encrypted before framing. The key byte is rotated by the position offset, so an identical plaintext byte encrypts to a different ciphertext byte at each position and simple frequency analysis does not apply.

### 📊 2. Data Link layer (§4)

* **Frame encapsulation (§4.1).** Header (8-bit sequence number, 8-bit type, 16-bit header CRC), payload, and a 32-bit CRC trailer, wrapped in `01111110` flags.
* **Bit stuffing (§4.1).** A `0` is inserted after five consecutive `1`s, so the body can never contain the flag pattern.
* **Frame delimiting at the receiver (§4.1).** The receiver is handed a *bit stream*, not a pre-cut frame, and locates boundaries by searching for the flag. This is what makes bit stuffing load-bearing: because six consecutive ones cannot occur in a body, every flag found is a real boundary. A frame whose flag was destroyed is lost and recovered by the sender's timer, and the bits spent resynchronising are counted rather than silently discarded.
* **Forward error correction (§4.2).** Hamming (12, 8) on every byte block corrects any single-bit error and reports how many it repaired.
* **Burst error detection (§4.2).** CRC-32 over header and payload catches the multi-bit damage Hamming cannot repair. A separate CRC-16 protects the header alone, so a frame whose payload is unrecoverable can still be identified well enough to trigger a targeted NAK instead of waiting for a timeout.
* **Selective Repeat ARQ (§4.3).** Sliding window with an independent timer per frame, ACK, NAK, out-of-order buffering, in-order delivery, duplicate detection, and a bounded retry budget.

### 📡 3. Physical layer (§2, §3)

* **Wired line coding (§2).** B8ZS and HDB3, both with a configurable per-symbol error rate. The GUI reports the two properties these codes exist to control: cumulative DC balance and the longest run without a pulse (B8ZS bounds it to 7, HDB3 to 3).
* **Wireless modulation (§3.1).** BPSK and Gray-coded 16-QAM, with the in-phase and quadrature components carried separately and plotted on a constellation diagram.
* **AWGN model (§3.2).** Noise is calibrated in **Eb/N0**, energy per *information* bit, for both modulations. Quoting a shared Eb/N0 is what makes the comparison meaningful: at equal Eb/N0 16-QAM is measurably noisier than BPSK, which is the trade-off it makes for four bits per symbol.

### 🖥️ 4. Live dashboards

* **Live Transmission tab.** Sliding-window state, per-frame timers, in-flight frames on both lanes, the framing strip, transmit and receive waveforms, a 16-QAM constellation with symbol errors highlighted, and a metrics grid where each cell is tagged with the specification section it evidences.
* **Terminal Simulation tab.** A single run with the full ARQ event log.
* **Performance Analysis tab (§5.1).** Sweeps Eb/N0 against several window sizes and plots Throughput and Goodput.

---

## ✅ Implementation status

| Requirement | Status | Where it is demonstrated |
|---|---|---|
| Security layer (§5.3) | ✅ Complete | `security.py`; Encryption metric cell |
| Line coding B8ZS / HDB3 (§2) | ✅ Complete | Waveform panel; Line code health cell |
| Modulation BPSK / 16-QAM (§3.1) | ✅ Complete | Waveform panel with I/Q traces; constellation |
| AWGN channel in Eb/N0 (§3.2) | ✅ Complete | Channel cell; Performance tab |
| Framing and bit stuffing (§4.1) | ✅ Complete | Framing strip; Frame on wire cell |
| Frame delimiting by flag search (§4.1) | ✅ Complete | Flag search cell; `FRAMING` log lines |
| Hamming (12,8) FEC (§4.2) | ✅ Complete | Hamming FEC cell; `FEC` log lines |
| CRC-32 and header CRC-16 (§4.2) | ✅ Complete | CRC / header drops cell |
| Selective Repeat ARQ (§4.3) | ✅ Complete | Protocol panel; ARQ and timer cells |
| Throughput and Goodput (§5.1) | ✅ Complete | Metric cells; Performance tab plots |
| Regression tests | ✅ 34 passing | `tests/test_simulator.py` |

`report.md` records the analysis this work was based on, every bug that was found and fixed, and the reasoning behind the remaining design decisions.

---

## 📁 Repository structure

```directory
CN1-Course-Project/
│
├── src/
│   ├── __init__.py                # Package declaration
│   ├── security.py                # Rotational XOR encryption (§5.3)
│   ├── data_link_layer.py         # Framing, stuffing, delimiting, Hamming, CRC (§4.1, §4.2)
│   ├── physical_layer.py          # B8ZS/HDB3, BPSK/16-QAM, AWGN (§2, §3)
│   ├── arq_protocol.py            # Selective Repeat sliding window (§4.3)
│   ├── performance_analysis.py    # Eb/N0 and window sweep (§5.1)
│   ├── gui.py                     # Tkinter dashboard, three tabs
│   └── main.py                    # Entry point and GUI launcher
│
├── tests/
│   └── test_simulator.py          # 34 regression tests across all layers
│
├── docs/
│   ├── Project_report.pdf         # Project report
│   └── Project_requirements.pdf   # Project requirements
│
├── run.py                         # Launches the GUI
├── requirements.txt               # Pinned dependencies
└── README.md                      # This file
```

---

## 🛠️ Installation and execution

### 1. Prerequisites

Python 3.8 or newer, plus the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the simulator

```bash
python run.py
```

**Live Transmission tab.** Type a message and a key, pick the channel and mode, then pres Start. Only the impairment control that applies is enabled: **Eb/N0** for a wireless channel and **Wired BER** for a wired one. Worthwhile things to try:

* Set the channel to *wireless / 16-QAM* to make the constellation panel appear. Lower Eb/N0 until received symbols cross the decision boundaries and turn red.
* Set *wired / B8ZS* with a BER of `0.000`, then raise it to `0.05`. Watch the Flag search cell begin to resynchronise and frames start failing CRC.
* Compare *B8ZS* and *HDB3* in the Line code health cell: the longest run without a pulse stays at or below 7 for B8ZS and 3 for HDB3.
* Shrink the window size to 1 to see the protocol degrade to stop-and-wait.

**Terminal Simulation tab.** One run, full ARQ log, integrity check against the original text.

**Performance Analysis tab.** Choose an Eb/N0 range and a set of window sizes to plot Throughput and Goodput.

### 3. Run the tests

```bash
python -m pytest -q
```

---

## 📐 Reference bit error rates

For BPSK and Gray-coded 16-QAM, with noise quoted as energy per information bit:

* **BPSK:**
  $$P_b = Q\left(\sqrt{2 E_b/N_0}\right) = \tfrac{1}{2}\,\mathrm{erfc}\left(\sqrt{E_b/N_0}\right)$$

* **16-QAM (approximate):**
  $$P_b \approx \tfrac{3}{8}\,\mathrm{erfc}\left(\sqrt{\tfrac{2}{5} E_b/N_0}\right)$$

The 16-QAM expression carries the penalty that makes the two comparable: four bits share a symbol, and the decision regions are correspondingly tighter.

---

## 📜 License

Licensed under the MIT License.
