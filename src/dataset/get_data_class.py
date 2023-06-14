import os
import base64
import numpy as np
from dataset.encryption import *
from secrets import token_bytes
from random import getrandbits
from multiprocessing import Pool
import re

"""
Dataset generation pipeline. No public facing functions here.
The functions needed for datasets.py.
"""

MAX_LENGTH = 16


#################
## RAW DATASET ##
#################

ENCODING = "ascii"
LARGE_DATASET = "dataset/assets/"
SMALL_DATASET = "dataset/assets2/"


def get_files(directory: str) -> list: 
    """ Retrieves filenames from the given directory
        
        Parameters:
            directory (str): the directory containing the .txt 
                            files to create the dataset
        Returns:
            (list): the list of filenames for constituing the dataset
    """

    files = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            files.append(f)
    return files


def load_txt(filename: str):
    """ Loads text from .txt file into usable list of strings.
        If an error occurs (due to encoding format), returns an empty array.
        
        Parameters:
            filename (str): the .txt file from which to retrieve strings
        Returns:
            (np.array): strings
    """
    with open(filename, mode="r", encoding=ENCODING) as f:
        try:
            data = f.read()
        except:
            print("Error: " + filename)
            return np.array([])
    
    return split_txt(data)

# S-AES is only for 16 bit plaintext/key
def load_txt_s_aes(filename: str):
    """ Loads text from .txt file into usable list of strings.
        If an error occurs (due to encoding format), returns an empty array.
        Only for S-AES, as it is only 16 bit plaintexts/keys.
        
        Parameters:
            filename (str): the .txt file from which to retrieve strings
        Returns:
            (np.array): strings
    """
    with open(filename, mode="r", encoding=ENCODING) as f:
        try:
            data = f.read()
        except:
            print("Error: " + filename)
            return np.array([])
    
    return split_txt_fixed_cut_n(data, 2)


"""
Some customization for the dataset can go here.
"""

def split_txt(data: str):
    """ Chosen splitting strategy.
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    return split_txt_fixed_cut(data)

def split_txt_sentence(data: str):
    """ Splitting strategy: cut by sentence.
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    d = data.replace('?', '.').replace('!', '.').replace('\n\n', '.').replace('\n-', '').replace('\n', ' ').replace(';','.').split(' ')
    return np.array([i.strip(' ')[:MAX_LENGTH] for i in d])


def split_txt_fixed_cut(data: str):
    """ Splitting strategy: cut every MAX_LENGTH byte.
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    return np.array([data[i:i+MAX_LENGTH] for i in range(0, len(data), MAX_LENGTH)])

def split_txt_fixed_cut_n(data: str, n: int):
    """ Splitting strategy: cut every n byte.
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    return np.array([data[i:i+n] for i in range(0, len(data), n)])



def split_txt_word(data: str):
    """ Splitting strategy: cut every word.
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    d = data.replace(':', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').replace('\n\n', ' ').replace('\n-', '').replace('\n', ' ').replace(';',' ').replace('.', ' ').split(' ')
    return np.array([i.strip(' ')[:MAX_LENGTH] for i in d])

def split_txt_group_of_words(data: str):
    """ Splitting strategy: cut every group of word (so that we get full words).
        
        Parameters:
            data (str): the array of strings to split
        Returns:
            (np.array): the list of data samples
    """
    d = data.replace('\n\n', '.').replace('\n-', '').replace('\n', ' ')
    return np.array([words.strip(' ')[:MAX_LENGTH] for words in re.split(r'[^\w\s]', d)])


# 
# returns a numpy array of strings
def string_dataset(directory: str):
    """ Creates a dataset of strings from directory of .txt files.
        
        Parameters:
            directory (str): sepcified directory with files
        Returns:
            (np.array): array of arrays of strings
    """
    dataset = np.array([])
    files = get_files(directory)
    for f in files:
        dataset = np.append(dataset, load_txt(f))
    return dataset

def string_dataset_simplified_aes(directory: str):
    """ Creates a dataset of strings from directory of .txt files for S-AES.
        
        Parameters:
            directory (str): sepcified directory with files
        Returns:
            (np.array): array of arrays of strings
    """
    dataset = np.array([])
    files = get_files(directory)
    for f in files:
        dataset = np.append(dataset, load_txt_s_aes(f))
    return dataset

############
### KEYS ###
############

def random_AES_key(key_size: int) -> bytes:
    """ Creates a random AES key of the specified size.
        
        Parameters:
            key_size (int): the desired size for the key (in bytes)
        Returns:
            (bytes): a random key in bytes
    """
    return token_bytes(key_size//8)


def random_SPECK_key(key_size: int) -> int:
    """ Creates a random SPECK key of the specified size.
        
        Parameters:
            key_size (int): the desired size for the key (in bytes)
        Returns:
            (int): a random key (as an integer)
    """
    return getrandbits(key_size)

def random_S_AES_key() -> int:
    """ Creates a random S-AES key of 16 bits (2 bytes).
        
        Parameters:
            key_size (int): the desired size for the key (in bytes)
        Returns:
            (int): a random key (as an integer)
    """
    return getrandbits(16)

##########################
### CIPHERTEXT DATASET ###
##########################


def aes_ciphertext_dataset_fixed_key(plaintext_dataset, key):
    """ Get corresponding ciphertexts for given plaintexts and fixed key,
        encrpyted with AES.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            key (bytes): the encryption key
        Returns:
            (np.array): an array of ciphertexts
    """
    
    keys = np.array([key for _ in range(len(plaintext_dataset))])

    return parallelize(list(zip(plaintext_dataset,keys)), aes_enc)


def aes_ciphertext_dataset_keys(plaintext_dataset, keys):
    """ Get corresponding ciphertexts for given plaintexts and list of keys,
        encrypted with AES.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            keys (np.array): an array of keys
        Returns:
            (np.array): an array of ciphertexts
    """
    
    return parallelize(list(zip(plaintext_dataset, keys)), aes_enc)


def speck_ciphertext_dataset_fixed_key(plaintext_dataset, key, key_size, block_size):
    """ Get corresponding ciphertexts for given plaintexts and fixed key,
        encrpyted with SPECK.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            key (int): the encryption key
            key_size (int): size of the key in bits
            block_size (int): block size of the cipher in bits
        Returns:
            (np.array): an array of ciphertexts
    """
    keys = np.array([key for _ in range(len(plaintext_dataset))])

    if (key_size == 64 and block_size == 32):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_32_64_enc)
    if (key_size == 128 and block_size == 64):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_64_128_enc)
    if (key_size == 128 and block_size == 128):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_128_128_enc)




def speck_ciphertext_dataset_keys(plaintext_dataset, keys, key_size, block_size):
    """ Get corresponding ciphertexts for given plaintexts and multiple keys,
        encrpyted with SPECK.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            keys (np.array): an array of keys
            key_size (int): size of the key in bits
            block_size (int): block size of the cipher in bits
        Returns:
            (np.array): an array of ciphertexts
    """
    if (key_size == 64 and block_size == 32):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_32_64_enc)
    if (key_size == 128 and block_size == 64):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_64_128_enc)
    if (key_size == 128 and block_size == 128):
        return parallelize(list(zip(plaintext_dataset, keys)), speck_128_128_enc)

def simplified_aes_dataset_fixed_key(plaintext_dataset, key):
    """ Get corresponding ciphertexts for given plaintexts and fixed key,
        encrpyted with Simplified AES.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            key (int): the encryption key
        Returns:
            (np.array): an array of ciphertexts
    """

    keys = np.array([key for _ in range(len(plaintext_dataset))])

    return parallelize(list(zip(plaintext_dataset, keys)), simplified_aes_enc)

def simplified_aes_dataset_keys(plaintext_dataset, keys):
    """ Get corresponding ciphertexts for given plaintexts and multiple keys,
        encrpyted with Simplified AES.
        
        Parameters:
            plaintext_dataset (np.array): the list of plaintexts
            keys (np.array): an array of keys
        Returns:
            (np.array): an array of ciphertexts
    """
    return parallelize(list(zip(plaintext_dataset, keys)), simplified_aes_enc)

##################
### Encryption ###
##################

def speck_32_64_enc(pt_key):
    """ Encrypt a plaintext with a key with Speck32/64.
        
        Parameters:
            pt_key (str, int): plaintext and key pair
        Returns:
            (np.array): an array of bits representing the encryption of the plaintext under the key
    """
    pt, key = pt_key
    pt = str_to_int(pt)
    ct = SPECK_32_64_encrypt(int(key), pt)
    return int_to_bits_32_pad(ct)

def speck_64_128_enc(pt_key):
    """ Encrypt a plaintext with a key with Speck64/128.
        
        Parameters:
            pt_key (str, int): plaintext and key pair
        Returns:
            (np.array): an array of bits representing the encryption of the plaintext under the key
    """
    pt, key = pt_key
    pt = str_to_int(pt)
    ct = SPECK_64_128_encrypt(int(key), pt)
    return int_to_bits_64_pad(ct)

def speck_128_128_enc(pt_key):
    """ Encrypt a plaintext with a key with Speck128/128.
        
        Parameters:
            pt_key (str, int): plaintext and key pair
        Returns:
            (np.array): an array of bits representing the encryption of the plaintext under the key
    """
    pt, key = pt_key
    pt = str_to_int(pt)
    ct = SPECK_128_128_encrypt(int(key), pt)
    return int_to_bits_128_pad(ct)

def aes_enc(pt_key):
    """ Encrypt a plaintext with a key with AES.
        
        Parameters:
            pt_key (str, bytes): plaintext and key pair
        Returns:
            (np.array): an array of bits representing the encryption of the plaintext under the key
    """
    pt, key = pt_key
    ct = AES_encrypt(key, str_to_bytes(pt))
    return bytes_to_bits(pad(ct))

def simplified_aes_enc(pt_key):
    """ Encrypt a plaintext with a key with S-AES.
        
        Parameters:
            pt_key (str, int): plaintext and key pair
        Returns:
            (np.array): an array of bits representing the encryption of the plaintext under the key
    """
    pt, key = pt_key
    pt = str_to_int(pt)
    ct = SAES_encrpyt(key, pt)
    return int_to_bits_16_pad(ct)


#################################
### Interface for datasets.py ###
#################################


def get_aes_dataset(dataset, key, split):
    """ Generate the pre-processed dataset for plaintext recovery against AES.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            key (bytes): the fixed key
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    labels_str = string_dataset(dataset)
    samples = aes_ciphertext_dataset_fixed_key(labels_str, key)

    labels = parallelize(labels_str, str_to_bits)

    split_at = (int) (split * np.shape(labels)[0])

    train_labels = np.array(labels[:split_at])
    test_labels = np.array(labels[split_at+1:])

    train_samples = np.array(samples[:split_at])
    test_samples = np.array(samples[split_at+1:])

    return train_labels, train_samples, test_labels, test_samples


def get_aes_dataset_key_recovery(dataset, key_size, split):
    """ Generate the pre-processed dataset for key recovery against AES.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            key_size (int): the size of the keys
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    plaintexts_str = string_dataset(dataset)
    keys = [random_AES_key(key_size) for _ in range(len(plaintexts_str))]

    ciphertexts = aes_ciphertext_dataset_keys(plaintexts_str, keys)
    plaintexts = parallelize(plaintexts_str, str_to_bits)

    # plaintext-ciphertext pairs
    samples = parallelize(list(zip(plaintexts, ciphertexts)), sampleify_pt_ct)

    # keys
    labels = parallelize(keys, bytes_to_bits)

    split_at = (int) (split * np.shape(labels)[0])

    train_labels = np.array(labels[:split_at])
    test_labels = np.array(labels[split_at+1:])

    train_samples = np.array(samples[:split_at])
    test_samples = np.array(samples[split_at+1:])

    return train_labels, train_samples, test_labels, test_samples


def get_speck_dataset(dataset, key: int, split, key_size, block_size):
    """ Generate the pre-processed dataset for plaintext recovery against Speck.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            key (int): the fixed key
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
            key_size (int): the size of the key
            block_size (int): the block size of the Speck cipher
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    labels_str = string_dataset(dataset)

    samples = speck_ciphertext_dataset_fixed_key(labels_str, key, key_size, block_size)

    labels = parallelize(labels_str, str_to_bits)

    split_at = (int) (split * np.shape(labels)[0])

    train_labels = np.array(labels[:split_at])
    test_labels = np.array(labels[split_at+1:])

    train_samples = np.array(samples[:split_at])
    test_samples = np.array(samples[split_at+1:])

    return train_labels, train_samples, test_labels, test_samples

def get_speck_dataset_key_recovery(dataset, split, key_size, block_size):
    """ Generate the pre-processed dataset for key recovery against Speck.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
            key_size (int): the size of the key
            block_size (int): the block size of the Speck cipher
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    plaintexts_str = string_dataset(dataset)
    keys = [random_SPECK_key(key_size) for _ in range(len(plaintexts_str))]

    ciphertexts = speck_ciphertext_dataset_keys(plaintexts_str, keys, key_size, block_size)

    plaintexts = parallelize(plaintexts_str, str_to_bits)

    samples = parallelize(list(zip(plaintexts, ciphertexts)), sampleify_pt_ct)

    # keys
    labels = parallelize(keys, int_to_bits)

    split_at = (int) (split * len(labels))

    train_labels = np.array(labels[:split_at], dtype=object)
    test_labels = np.array(labels[split_at+1:], dtype=object)

    train_samples = np.array(samples[:split_at], dtype=object)
    test_samples = np.array(samples[split_at+1:], dtype=object)

    return train_labels, train_samples, test_labels, test_samples

def get_simplified_aes_dataset(dataset, key: int, split):
    """ Generate the pre-processed dataset for plaintext recovery against Simplified AES.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            key (int): the fixed key
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    labels_str = string_dataset_simplified_aes(dataset)

    samples = simplified_aes_dataset_fixed_key(labels_str, key)

    labels = parallelize(labels_str, str_to_bits_16_pad)

    split_at = (int) (split * np.shape(labels)[0])

    train_labels = np.array(labels[:split_at])
    test_labels = np.array(labels[split_at+1:])

    train_samples = np.array(samples[:split_at])
    test_samples = np.array(samples[split_at+1:])

    return train_labels, train_samples, test_labels, test_samples

def get_simplified_aes_dataset_key_recovery(dataset, split):
    """ Generate the pre-processed dataset for key recovery against Simplified AES.
        
        Parameters:
            dataset (str): the directory with the .txt files to form a dataset
            split (float): the proportion of training samples compared to testing samples (between 0 and 1)
        Returns:
            train_samples, train_labels, test_samples, test_labels (tuple): train/test samples/labels
    """
    plaintexts_str = string_dataset_simplified_aes(dataset)

    keys = [random_S_AES_key() for _ in range(len(plaintexts_str))]

    ciphertexts = simplified_aes_dataset_keys(plaintexts_str, keys)
    plaintexts = parallelize(plaintexts_str, str_to_bits_16_pad)

    samples = parallelize(list(zip(plaintexts, ciphertexts)), sampleify_pt_ct)

    labels = parallelize(keys, int_to_bits_16_pad)

    split_at = (int) (split * np.shape(labels)[0])

    train_labels = np.array(labels[:split_at])
    test_labels = np.array(labels[split_at+1:])

    train_samples = np.array(samples[:split_at], dtype=object)
    test_samples = np.array(samples[split_at+1:], dtype=object)

    return train_labels, train_samples, test_labels, test_samples



###################################
### Type Converters and Helpers ###
###################################


def bytes_to_bits(b):
    """ Transforms an array of bytes to a numpy array of 0's and 1's
        
        Parameters:
            b (bytes)
        Returns:
            (np.array): array of 1's and 0's
    """
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def str_to_bits(data):
    """ Transforms a string to a numpy array of 0's and 1's
        
        Parameters:
            data (str)
        Returns:
            (np.array): array of 1's and 0's
    """
    str_bytes = data.encode(encoding=ENCODING)
    return bytes_to_bits(pad(str_bytes))

def str_to_bits_16_pad(data):
    """ Transforms a string to a numpy array of 0's and 1's with padding
        up to 16 bits
        
        Parameters:
            data (str)
        Returns:
            (np.array): array of 1's and 0's
    """
    str_bytes = data.encode(encoding=ENCODING)
    return bytes_to_bits(pad_n(str_bytes, 2))

def int_to_bits(input: int):
    """ Transforms an integer to a numpy array of 0's and 1's
        
        Parameters:
            input (int)
        Returns:
            (np.array): array of 1's and 0's
    """
    return bytes_to_bits(int_to_bytes(input))

# Note: all these functions were not coded with a parametrized n 
# to be able to parallelize them.

def int_to_bits_16_pad(input: int):
    """ Transforms an integer to a numpy array of 0's and 1's with padding
        up to 16 bits
        
        Parameters:
            input (int)
        Returns:
            (np.array): array of 1's and 0's
    """
    return bytes_to_bits(pad_n(int_to_bytes(input), 2))

def int_to_bits_32_pad(input: int):
    """ Transforms an integer to a numpy array of 0's and 1's with padding
        up to 32 bits
        
        Parameters:
            input (int)
        Returns:
            (np.array): array of 1's and 0's
    """
    return bytes_to_bits(pad_n(int_to_bytes(input), 4))

def int_to_bits_64_pad(input: int):
    """ Transforms an integer to a numpy array of 0's and 1's with padding
        up to 64 bits
        
        Parameters:
            input (int)
        Returns:
            (np.array): array of 1's and 0's
    """
    return bytes_to_bits(pad_n(int_to_bytes(input), 8))

def int_to_bits_128_pad(input: int):
    """ Transforms an integer to a numpy array of 0's and 1's with padding
        up to 128 bits
        
        Parameters:
            input (int)
        Returns:
            (np.array): array of 1's and 0's
    """
    return bytes_to_bits(pad_n(int_to_bytes(input), 16))

def pad(b):
    """ Adds all 0 bytes for b to be of length MAX_LENGTH
        
        Parameters:
            b (bytes)
        Returns:
            (bytes): padded bytes with '\x00' padding
    """
    l = len(b)
    if l < MAX_LENGTH:
        new_b = bytes(b) + (MAX_LENGTH - l) * b'\x00'
        return new_b
    return b[:MAX_LENGTH]

def pad_n(b, n):
    """ Adds all 0 bytes for b to be of length n
        
        Parameters:
            b (bytes)
            n (int): how much to pad
        Returns:
            (bytes): padded bytes with '\x00' padding
    """
    l = len(b)
    if l < n:
        new_b = bytes(b) + (n - l) * b'\x00'
        return new_b
    return b[:n]

def str_to_bytes(input: str) -> bytes:
    """ Transforms a string to bytes
        
        Parameters:
            input (str)
        Returns:
            (bytes): encoded string as bytes
    """
    return input.encode(ENCODING)

def bytes_to_str(input: bytes) -> str:
    """ Transforms bytes to a string
        
        Parameters:
            input (bytes)
        Returns:
            (str): decoded bytes as string
    """
    return input.decode(ENCODING)

def str_to_int(input: str) -> int:
    """ Transforms a string to an integer
        
        Parameters:
            input (str)
        Returns:
            (int)
    """
    return int.from_bytes(str_to_bytes(input))

def int_to_str(input: int) -> str:
    """ Transforms an integer to a string
        
        Parameters:
            input (int)
        Returns:
            (str)
    """
    inter = int_to_bytes(input)
    return bytes_to_str(base64.b64encode(inter))

def int_to_bytes(input: int) -> bytes:
    """ Transforms an integer to bytes
        
        Parameters:
            input (int)
        Returns:
            (bytes)
    """
    return int(input).to_bytes((int(input).bit_length() + 7) // 8)

def bytes_to_int(input: bytes) -> int:
    """ Transforms bytes to an integer
        
        Parameters:
            input (bytes)
        Returns:
            (int)
    """
    return int.from_bytes(input)

def sampleify_pt_ct(pair):
    """ Concatenates a plaintext and a ciphertext to form a sample (key recovery attack).
        
        Parameters:
            pair (tuple): plaintext-ciphertext pair
        Returns:
            (np.array): the pair, concatenated
    """
    return np.concatenate(pair)

################
### Parallel ###
################

def parallelize(data, function):
    """ Makes a function that run in parallel with elements of the list as arguments
        
        Parameters:
            data (list): the list of arguments
            function (func): the function to run in parallel
        Returns:
            (list): the list of results
    """
    with Pool() as pool:
        results = pool.map(function, data)
    return results
