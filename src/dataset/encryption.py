from Crypto.Cipher import AES
from base64 import b64encode
from Crypto.Util.Padding import pad
from dataset.saes import SimplifiedAES
from speck import SpeckCipher

"""
encryption.py contains methods to encrypt plaintexts.

References:
- https://pypi.org/project/simonspeckciphers/
- https://pycryptodome.readthedocs.io/en/latest/src/cipher/classic.html#ecb-mode
"""

def AES_encrypt(key: bytes, plaintext: bytes) -> bytes:
    """ Encrypts a plaintext with the given key using AES
        in CBC mode.
        
        Parameters:
            key (bytes): the encryption key
            plaintext (bytes): the plaintext to be encrypted
        Returns:
            ciphertext (bytes): the ciphertext
    """
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.encrypt(pad(plaintext, AES.block_size))

def SAES_encrpyt(key: int, plaintext: int) -> int:
    """ Encrypts a plaintext with the given key using Simmplified AES.
        
        Parameters:
            key (int): the encryption key
            plaintext (int): the plaintext to be encrypted
        Returns:
            ciphertext (int): the ciphertext
    """
    return SimplifiedAES(key).encrypt(plaintext)


# block size/key size: 32/64, 64/128, 128/128
# key_int = int.from_bytes(key_bytes, byteorder='big', signed='False')
def SPECK_32_64_encrypt(key: int, plaintext: int) -> int:
    """ Encrypts a plaintext with the given key using Speck 32/64 in ECB mode.
        
        Parameters:
            key (int): the encryption key
            plaintext (int): the plaintext to be encrypted
        Returns:
            ciphertext (int): the ciphertext
    """
    cipher = SpeckCipher(key, mode='ECB', key_size=64, block_size=32)
    return cipher.encrypt(plaintext)

def SPECK_64_128_encrypt(key, plaintext):
    """ Encrypts a plaintext with the given key using Speck 64/128 in ECB mode.
        
        Parameters:
            key (int): the encryption key
            plaintext (int): the plaintext to be encrypted
        Returns:
            ciphertext (int): the ciphertext
    """
    cipher = SpeckCipher(key, mode='ECB', key_size=128, block_size=64)
    return cipher.encrypt(plaintext)

def SPECK_128_128_encrypt(key, plaintext):
    """ Encrypts a plaintext with the given key using Speck 128/128 in ECB mode.
        
        Parameters:
            key (int): the encryption key
            plaintext (int): the plaintext to be encrypted
        Returns:
            ciphertext (int): the ciphertext
    """
    cipher = SpeckCipher(key, mode='ECB', key_size=128, block_size=128)
    return cipher.encrypt(plaintext)


def SPECK_encrypt(key, plaintext, key_size, block_size):
    """ Encrypts a plaintext with the given key using Speck 128/128 in ECB mode,
        with a specified key and block size.
        Parameters:
            key (int): the encryption key
            plaintext (int): the plaintext to be encrypted
            key_size (int): the size of the key
            block_size (int): the block size of the cipher
        Returns:
            ciphertext (int): the ciphertext
    """
    # assume correct usage (valid key sizes and block sizes)
    cipher = SpeckCipher(key, mode='ECB', key_size=key_size, block_size=block_size)
    return cipher.encrypt(plaintext)
