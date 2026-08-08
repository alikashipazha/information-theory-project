"""
Security & Cryptography Module
Provides custom rotational XOR encryption and decryption layers with a dynamic key 
to simulate secure data communication over physical channels.
"""

class SecurityLayer:
    @staticmethod
    def encrypt_xor_rotational(data_bytes: bytes, key_str: str) -> bytes:
        """
        Symmetrically encrypts input data using a rotational XOR cipher.
        The key is dynamically altered based on the byte index to eliminate 
        frequency patterns, making eavesdropping on the physical layer ineffective.
        """
        encrypted = bytearray()
        key_bytes = key_str.encode('utf-8')
        key_len = len(key_bytes)
        if key_len == 0:
            return data_bytes
            
        for i, b in enumerate(data_bytes):
            dynamic_key_byte = key_bytes[i % key_len] ^ (i & 0xFF)
            encrypted.append(b ^ dynamic_key_byte)
        return bytes(encrypted)

    @staticmethod
    def decrypt_xor_rotational(encrypted_bytes: bytes, key_str: str) -> bytes:
        """
        Symmetrically decrypts the data. Since XOR is self-inverting,
        decryption uses the exact same algorithm.
        """
        return SecurityLayer.encrypt_xor_rotational(encrypted_bytes, key_str)
