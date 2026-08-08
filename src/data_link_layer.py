"""
Data Link Layer Module
Provides error correction (Hamming), error detection (CRC-32), and framing (Bit-Stuffing).
"""

import binascii
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

class DataLinkUtils:
    @staticmethod
    def hamming_12_8_encode(bits: List[int]) -> List[int]:
        """Encodes 8 bits of data to a 12-bit Hamming code block (SEC)."""
        p = [0] * 13
        p[3] = bits[0]; p[5] = bits[1]; p[6] = bits[2]; p[7] = bits[3]
        p[9] = bits[4]; p[10] = bits[5]; p[11] = bits[6]; p[12] = bits[7]
        
        p[1] = p[3] ^ p[5] ^ p[7] ^ p[9] ^ p[11]
        p[2] = p[3] ^ p[6] ^ p[7] ^ p[10] ^ p[11]
        p[4] = p[5] ^ p[6] ^ p[7] ^ p[12]
        p[8] = p[9] ^ p[10] ^ p[11] ^ p[12]
        return p[1:13]

    @staticmethod
    def hamming_12_8_decode_ex(code_bits: List[int]) -> Tuple[List[int], bool]:
        """
        Decodes a 12-bit Hamming block back to 8 bits, correcting single-bit errors.
        Also reports whether a correction was actually applied, so the upper layers can
        expose Forward Error Correction activity.
        """
        p = [0] + list(code_bits)
        c1 = p[1] ^ p[3] ^ p[5] ^ p[7] ^ p[9] ^ p[11]
        c2 = p[2] ^ p[3] ^ p[6] ^ p[7] ^ p[10] ^ p[11]
        c4 = p[4] ^ p[5] ^ p[6] ^ p[7] ^ p[12]
        c8 = p[8] ^ p[9] ^ p[10] ^ p[11] ^ p[12]
        syndrome = c1 + (c2 << 1) + (c4 << 2) + (c8 << 3)
        corrected = 1 <= syndrome <= 12
        if corrected:
            p[syndrome] ^= 1
        return [p[3], p[5], p[6], p[7], p[9], p[10], p[11], p[12]], corrected

    @staticmethod
    def hamming_12_8_decode(code_bits: List[int]) -> List[int]:
        """Decodes a 12-bit Hamming block back to 8 bits, correcting single-bit errors."""
        data_bits, _ = DataLinkUtils.hamming_12_8_decode_ex(code_bits)
        return data_bits

    @staticmethod
    def bits_to_bytes(bits: List[int]) -> bytes:
        b = bytearray()
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) < 8:
                byte_bits = byte_bits + [0] * (8 - len(byte_bits))
            val = 0
            for bit in byte_bits:
                val = (val << 1) | bit
            b.append(val)
        return bytes(b)

    @staticmethod
    def bytes_to_bits(data_bytes: bytes) -> List[int]:
        bits = []
        for byte in data_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def calculate_crc32(bits: List[int]) -> List[int]:
        """Computes standard CRC-32 checksum and returns its 32-bit representation."""
        data_bytes = DataLinkUtils.bits_to_bytes(bits)
        crc = binascii.crc32(data_bytes) & 0xffffffff
        crc_bits = []
        for i in range(31, -1, -1):
            crc_bits.append((crc >> i) & 1)
        return crc_bits

    @staticmethod
    def calculate_crc16(bits: List[int]) -> List[int]:
        """
        Computes a CRC-16/CCITT-FALSE checksum used to protect the frame header on its own.
        A separately verifiable header lets the receiver still trust the sequence number of a
        frame whose payload failed CRC-32, which is what makes a targeted NAK possible.
        """
        crc = 0xFFFF
        for byte in DataLinkUtils.bits_to_bytes(bits):
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        return [(crc >> i) & 1 for i in range(15, -1, -1)]

    @staticmethod
    def bit_stuff_ex(bits: List[int]) -> Tuple[List[int], List[int]]:
        """
        Stuffs a '0' after five consecutive '1' bits so the body can never contain the flag.

        Also returns the indices, in the stuffed stream, where each inserted bit landed. The
        GUI uses those positions to point at the stuffing on the wire, which is the only way
        to show that this step is doing anything.
        """
        stuffed: List[int] = []
        positions: List[int] = []
        consecutive_ones = 0
        for bit in bits:
            stuffed.append(bit)
            if bit == 1:
                consecutive_ones += 1
                if consecutive_ones == 5:
                    positions.append(len(stuffed))
                    stuffed.append(0)
                    consecutive_ones = 0
            else:
                consecutive_ones = 0
        return stuffed, positions

    @staticmethod
    def bit_stuff(bits: List[int]) -> List[int]:
        return DataLinkUtils.bit_stuff_ex(bits)[0]

    @staticmethod
    def bit_unstuff(stuffed_bits: List[int]) -> List[int]:
        """
        Removes the '0' that the stuffer inserted after every five consecutive '1' bits.

        The removal is unconditional, and that is the point. The earlier version only
        removed the following bit when it was actually 0, so noise flipping a stuff bit to 1
        left an extra bit in the stream and shifted everything after it: a single bit error
        destroyed the whole frame's alignment. Removing unconditionally keeps the frame
        aligned and leaves a localised error that Hamming and the CRCs can act on.
        """
        unstuffed: List[int] = []
        consecutive_ones = 0
        i = 0
        total = len(stuffed_bits)
        while i < total:
            bit = stuffed_bits[i]
            unstuffed.append(bit)
            i += 1
            if bit == 1:
                consecutive_ones += 1
                if consecutive_ones == 5:
                    i += 1  # drop the stuff bit whatever value it arrived with
                    consecutive_ones = 0
            else:
                consecutive_ones = 0
        return unstuffed


@dataclass
class FrameEncoding:
    """One frame as it goes on the wire, with the framing detail the GUI needs to show."""

    bits: List[int]
    stuffed_positions: List[int]
    flag_bits: int
    body_bits: int

    @property
    def stuffed_count(self) -> int:
        return len(self.stuffed_positions)


@dataclass
class FrameDecodeResult:
    """
    Outcome of decoding one received frame.

    The two validity flags are deliberately independent: ``header_valid`` without
    ``payload_valid`` means the frame was damaged but is still identifiable, which is the
    case where the receiver answers with a NAK instead of waiting for a sender timeout.
    """
    header_valid: bool = False
    payload_valid: bool = False
    seq_num: Optional[int] = None
    frame_type: Optional[str] = None
    payload: bytes = b""
    corrected_bits: int = 0
    error: Optional[str] = None

    # True when Hamming altered a block that carries header bits. A correction is only a
    # repair if exactly one bit was damaged; with two damaged bits the syndrome points at an
    # innocent third bit and the "repair" invents a header. Callers that cannot afford to act
    # on a wrong sequence number use this to decide how much to trust the header.
    header_corrected: bool = False

    def as_frame(self) -> Optional['Frame']:
        if not (self.header_valid and self.payload_valid):
            return None
        return Frame(self.seq_num, self.frame_type, self.payload)


class Frame:
    FLAG = [0, 1, 1, 1, 1, 1, 1, 0]  # Frame boundary flag (0x7E)
    TYPE_CODES = {'DATA': 0, 'ACK': 1, 'NAK': 2}
    TYPE_NAMES = {v: k for k, v in TYPE_CODES.items()}

    # Header = 8-bit sequence + 8-bit type + 16-bit header CRC; trailer = 32-bit CRC.
    HEADER_BITS = 32
    TRAILER_BITS = 32

    def __init__(self, seq_num: int, frame_type: str, payload: bytes):
        self.seq_num = seq_num
        self.frame_type = frame_type  # 'DATA', 'ACK', 'NAK'
        self.payload = payload

    def to_bits(self) -> List[int]:
        """Bits as they go on the wire; see :meth:`encode` for the framing detail."""
        return self.encode().bits

    def encode(self) -> FrameEncoding:
        """
        Encapsulates fields, appends CRC-32, applies Hamming encoding,
        performs bit-stuffing, and wraps with starting and ending flags.
        """
        type_val = self.TYPE_CODES.get(self.frame_type, 0)
        seq_bits = [(self.seq_num >> i) & 1 for i in range(7, -1, -1)]
        type_bits = [(type_val >> i) & 1 for i in range(7, -1, -1)]
        header_bits = seq_bits + type_bits
        header_bits += DataLinkUtils.calculate_crc16(header_bits)

        payload_bits = DataLinkUtils.bytes_to_bits(self.payload)
        header_payload = header_bits + payload_bits
        crc_bits = DataLinkUtils.calculate_crc32(header_payload)
        combined_bits = header_payload + crc_bits
        
        # Apply Hamming (12, 8) encoding to byte blocks
        hamming_encoded_bits = []
        for i in range(0, len(combined_bits), 8):
            block = combined_bits[i:i+8]
            if len(block) < 8:
                block = block + [0] * (8 - len(block))
            hamming_encoded_bits.extend(DataLinkUtils.hamming_12_8_encode(block))
            
        stuffed_bits, stuffed_positions = DataLinkUtils.bit_stuff_ex(hamming_encoded_bits)
        flag_len = len(self.FLAG)
        return FrameEncoding(
            bits=self.FLAG + stuffed_bits + self.FLAG,
            # Shift the positions past the opening flag so they index the wire stream.
            stuffed_positions=[p + flag_len for p in stuffed_positions],
            flag_bits=flag_len,
            body_bits=len(stuffed_bits),
        )

    @classmethod
    def decode(cls, bits: List[int]) -> FrameDecodeResult:
        """
        Reverses the transmit pipeline and reports exactly where the frame failed.

        Order matches the specification: unstuff, Hamming-correct single-bit errors, then
        verify the checksums. The header CRC is checked before the payload CRC so that a
        surviving header can still identify a frame whose payload is unrecoverable.
        """
        result = FrameDecodeResult()

        if len(bits) < 16 or bits[:8] != cls.FLAG or bits[-8:] != cls.FLAG:
            result.error = "FLAG"
            return result

        hamming_encoded_bits = DataLinkUtils.bit_unstuff(bits[8:-8])
        if len(hamming_encoded_bits) % 12 != 0:
            result.error = "ALIGNMENT"
            return result

        combined_bits = []
        for i in range(0, len(hamming_encoded_bits), 12):
            decoded_block, corrected = DataLinkUtils.hamming_12_8_decode_ex(hamming_encoded_bits[i:i+12])
            if corrected:
                result.corrected_bits += 1
                if len(combined_bits) < cls.HEADER_BITS:
                    result.header_corrected = True
            combined_bits.extend(decoded_block)

        if len(combined_bits) < cls.HEADER_BITS + cls.TRAILER_BITS:
            result.error = "TRUNCATED"
            return result

        header_bits = combined_bits[:cls.HEADER_BITS]
        seq_bits, type_bits, header_crc = header_bits[0:8], header_bits[8:16], header_bits[16:32]

        if DataLinkUtils.calculate_crc16(seq_bits + type_bits) != header_crc:
            result.error = "HEADER_CRC"
            return result

        type_val = int("".join(str(b) for b in type_bits), 2)
        if type_val not in cls.TYPE_NAMES:
            result.error = "HEADER_CRC"
            return result

        result.header_valid = True
        result.seq_num = int("".join(str(b) for b in seq_bits), 2)
        result.frame_type = cls.TYPE_NAMES[type_val]

        body = combined_bits[cls.HEADER_BITS:]
        payload_bits, received_crc = body[:-cls.TRAILER_BITS], body[-cls.TRAILER_BITS:]

        if DataLinkUtils.calculate_crc32(header_bits + payload_bits) != received_crc:
            result.error = "PAYLOAD_CRC"  # Residual burst error Hamming could not repair
            return result

        result.payload_valid = True
        result.payload = DataLinkUtils.bits_to_bytes(payload_bits)
        return result

    @classmethod
    def from_bits(cls, bits: List[int]) -> Optional['Frame']:
        """Returns a fully verified frame, or None if any integrity check failed."""
        return cls.decode(bits).as_frame()


class FrameDelimiter:
    """
    Recovers frame boundaries from a continuous bit stream (specification section 4.1).

    This is what makes bit stuffing load-bearing rather than decorative. Because the stuffer
    guarantees the body can never contain six consecutive ones, any occurrence of 01111110 in
    the stream is a genuine boundary and never data that happens to look like one, so the
    receiver can find frames without being told where they start.

    The delimiter is deliberately stateful: a frame split across two reads stays in the
    buffer until its closing flag arrives, and bits that precede a flag are counted as lost
    to resynchronisation instead of being discarded silently.
    """

    def __init__(self, max_frame_bits: int = 8192):
        self.buffer: List[int] = []
        self.max_frame_bits = max_frame_bits
        self.frames_found = 0
        self.resyncs = 0
        self.discarded_bits = 0

    def reset(self) -> None:
        self.buffer.clear()

    @property
    def flag_len(self) -> int:
        return len(Frame.FLAG)

    def _find_flag(self, start: int) -> Optional[int]:
        flag = Frame.FLAG
        limit = len(self.buffer) - self.flag_len
        for index in range(start, limit + 1):
            if self.buffer[index:index + self.flag_len] == flag:
                return index
        return None

    def _drop(self, count: int) -> None:
        if count > 0:
            del self.buffer[:count]

    def feed(self, bits: Sequence[int]) -> List[List[int]]:
        """Adds bits to the stream and returns every complete frame that became available."""
        self.buffer.extend(bits)
        frames: List[List[int]] = []

        while True:
            start = self._find_flag(0)

            if start is None:
                # No flag at all: keep only enough tail to complete one split across reads.
                stale = max(0, len(self.buffer) - (self.flag_len - 1))
                if stale:
                    self.discarded_bits += stale
                    self._drop(stale)
                break

            if start > 0:
                # Bits ahead of the opening flag cannot belong to a frame we can parse.
                self.resyncs += 1
                self.discarded_bits += start
                self._drop(start)
                continue

            end = self._find_flag(self.flag_len)
            if end is None:
                if len(self.buffer) > self.max_frame_bits:
                    # No closing flag within a plausible frame length: abandon this opening
                    # flag and hunt for the next one, the way a real receiver aborts an
                    # over-long frame instead of buffering for ever.
                    self.resyncs += 1
                    self.discarded_bits += self.flag_len
                    self._drop(self.flag_len)
                    continue
                break

            if end == self.flag_len:
                # Two flags back to back: a shared delimiter or idle fill, not a frame.
                self._drop(self.flag_len)
                continue

            frames.append(list(self.buffer[:end + self.flag_len]))
            self.frames_found += 1
            # Stop before the closing flag so it can serve as the next opening flag too,
            # which keeps both the doubled-flag and shared-flag conventions working.
            self._drop(end)

        return frames
