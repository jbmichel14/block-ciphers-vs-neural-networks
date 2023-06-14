from dataset.get_data_class import SMALL_DATASET, LARGE_DATASET, get_aes_dataset, get_aes_dataset_key_recovery, get_speck_dataset, get_speck_dataset_key_recovery, get_simplified_aes_dataset, get_simplified_aes_dataset_key_recovery

"""
Fixed AES key computed with 
`token_bytes(16)`
(from secrets import token_bytes)
"""
AES_KEY_128 = b'\xcc\x97\x84\xaf\xbe\xc43J\x1b\x83WzfEm\xcb'
AES_KEY_192 = b"\xda\x19\x8b\x8d\xc8'\x9d\xe1#\xa8\xdc\x15\x15(\r&\xae\t\x9f\xd7r\x0b\x9cG"
AES_KEY_256 = b"h\x8d=j\xce\x80\x9f\xc0\xb9)\xae\xb7\xd2\xa2\x01\x96\x99'\xcb\xfdK\x84\x07&\xca\xc0yDT\xe6Bd"

"""
Fixed S-AES key computed with random.getrandbits(16)
"""
S_AES_KEY_16 = 0b0001010011011100

"""
Fixed SPECK key computed with random.getrandbits(128) (import random)
"""
SPECK_KEY_128 = 150631321183738849573984615378888218990
SPECK_KEY_64 = 4735680038781233211

###############################
### Public Facing Interface ###
###############################

# for plaintext recovery
class AESDatasetCiphertextPlaintext():
    """
    A class representing a dataset for plaintext recovery against AES,
    with plaintexts as labels and ciphertexts as samples.

    Attributes
    ----------
    key_size (int): the size of the AES key, default is 128 bits
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples
    get_key: returns the fixed key

    """

    def __init__(self,
                 key_size: int = 128,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):

        self.key = aes_key(key_size)

        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_aes_dataset(
            self.dataset,
            self.key,
            self.split
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples
    
    def get_key(self):
        return self.key
    

# for key recovery
class AESDatasetCiphertextPlaintextPairKey():
    """
    A class representing a dataset for key recovery against AES,
    with keys as labels and plaintext-ciphertext pairs as samples.

    Attributes
    ----------
    key_size (int): the size of the AES key, default is 128 bits
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples

    """
    def __init__(self,
                 key_size: int = 128,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):

        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)

        self.key_size = check_aes_key_size(key_size)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_aes_dataset_key_recovery(
            self.dataset,
            self.key_size,
            self.split
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples
    

# for plaintext recovery
class SPECKDatasetCiphertextPlaintext():
    """
    A class representing a dataset for plaintext recovery against Speck,
    with plaintexts as labels and ciphertexts as samples.

    Attributes
    ----------
    key_size (int): the size of the Speck key, default is 128 bits
    block_size (int): the block size of the Speck cipher, default is 128 bits
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples
    get_key: returns the fixed key

    """
    def __init__(self,
                 key_size: int = 128,
                 block_size: int = 128,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):


        self.key_size, self.block_size = check_speck_key_size_block_size(key_size, block_size)
        self.key = speck_key(key_size)
        

        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_speck_dataset(
            self.dataset,
            self.key,
            self.split,
            self.key_size,
            self.block_size
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples
    
    def get_key(self):
        return self.key

# for key recovery
class SPECKDatasetCiphertextPlaintextPairKey():
    """
    A class representing a dataset for key recovery against Speck,
    with keys as labels and plaintext-ciphertext pairs as samples.

    Attributes
    ----------
    key_size (int): the size of the Speck key, default is 128 bits
    block_size (int): the block size of the Speck cipher, default is 128 bits
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples

    """
    def __init__(self,
                 key_size: int = 128,
                 block_size: int = 128,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):


        self.key_size, self.block_size = check_speck_key_size_block_size(key_size, block_size)

        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_speck_dataset_key_recovery(
            self.dataset,
            self.split,
            self.key_size,
            self.block_size
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples

# for plaintext recovery
class SimplifiedAESDatasetCiphertextPlaintext():
    """
    A class representing a dataset for plaintext recovery against S-AES,
    with plaintexts as labels and ciphertexts as samples.

    Attributes
    ----------
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples
    get_key: returns the fixed key

    """
    def __init__(self,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):


        self.key = S_AES_KEY_16


        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_simplified_aes_dataset(
            self.dataset,
            self.key,
            self.split,
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples
    
    def get_key(self):
        return self.key

# for key recovery
class SimplifiedAESDatasetCiphertextPlaintextPairKey():
    """
    A class representing a dataset for key recovery against S-AES,
    with keys as labels and plaintext-ciphertext pairs as samples.

    Attributes
    ----------
    dataset_size (str): specified dataset size (large or small), default is large
    split (float): the proportion of training samples compared to testing samples (between 0 and 1)

    Methods
    -------
    get_data: returns the dataset (train_samples, train_labels, test_samples, test_labels)
    get_train_labels: returns the train labels
    get_train_samples: returns the train samples
    get_test_labels: returns the test labels
    get_test_samples: returns the test_samples

    """

    def __init__(self,
                 dataset_size: str = 'large',
                 split: float = 0.7,
                 ):

        self.dataset = dataset_from_size(dataset_size)

        self.split = check_split(split)
        
        self.train_labels, self.train_samples, self.test_labels, self.test_samples = get_simplified_aes_dataset_key_recovery(
            self.dataset,
            self.split,
        )

    def get_data(self):
        return self.train_labels, self.train_samples, self.test_labels, self.test_samples
    def get_train_labels(self):
        return self.train_labels
    def get_train_samples(self):
        return self.train_samples
    def get_test_labels(self):
        return self.test_labels
    def get_test_samples(self):
        return self.test_samples
    
###############
### Helpers ###
###############

def aes_key(key_size):
    """ Returns a valid key (of valid length even if given key size 
        if invalid) for AES out of a key size. Valid sizes are 128, 192 or 256.
        
        Parameters:
            key_size (int): the provided key size
        Returns:
            key (bytes): the valid key in bytes
    """
    if key_size == 192:
        return AES_KEY_192
    elif key_size == 256:
        return AES_KEY_256
    else:
        return AES_KEY_128
    
def speck_key(key_size):
    """ Returns a valid key (of valid length even if given key size 
        if invalid) for Speck out of a key size. Valid sizes are 64 or 128.
        
        Parameters:
            key_size (int): the provided key size
        Returns:
            key (int): the valid key
    """

    if key_size == 64:
        return SPECK_KEY_64
    else:
        return SPECK_KEY_128

def check_aes_key_size(key_size):
    """ Returns a valid key size for Speck out of a key size. 
        Valid sizes are 64 or 128.
        
        Parameters:
            key_size (int): the provided key size
        Returns:
            (int): the valid key size
    """
    if key_size == 192:
        return 192
    elif key_size == 256:
        return 256
    else:
        return 128
    
def check_speck_key_size_block_size(key_size, block_size):
    """ Returns a valid key size and block size for Speck.
        Valid sizes are (64,32), (128,64) or (128, 128).
        
        Parameters:
            key_size (int): the provided key size
            block_size (int): the provided block size
        Returns:
            (int, inz): the valid key size and blocki size
    """
    if key_size == 64:
        return 64, 32
    else:
        if block_size == 64:
            return 128, 64
        else:
            return 128, 128

def dataset_from_size(size):
    """ Returns a asset folder out of a given dataset size
        
        Parameters:
            size (int): the provided dataset size (should be 'large' or 'small)
        Returns:
            folder (str): valid folder path for dataset
    """
    if size == 'small':
        return SMALL_DATASET
    else:
        return LARGE_DATASET
    
def check_split(split):
    """ Returns a valid split.
        
        Parameters:
            split (float): the given split
        Returns:
            (float): the valid split
    """
    if split < 0 or split > 1:
        return 0.7
    else:
        return split