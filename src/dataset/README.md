# Datasets for Round-Reduced Ciphers

## Source
The code in this submodule was implemented for the semester project _Attacking Pseudo-randomness with Deep Learning_ by Jean-Baptiste Michel.

It is useful for the experimentation on AES, S-AES and the Speck cipher.


## Overview

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Explanation                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| assets | This folder contains all text files for the large dataset. |
| assets 2 | This folder contains one text file for the small datase |
| datasets.py | Public facing interface with classes for each type of dataset (corresponding to an attack). |
| encryption.py | Encryption methods for AES and Speck |
| saes.py | Implementation of S-AES. Source: <https://github.com/mayank-02/simplified-aes> |
| get_data_class.py | Data generation and pre-processing code. |

## Usage

To import an AES plaintext recovery dataset, proceed as follows:

```
from datasets import AESDatasetCiphertextPlaintext

data = AESDatasetCiphertextPlaintext(128, 'small')

train_labels, train_samples, test_labels, test_samples = data.get_data()

```

For other attacks: 
- AES key recovery: `AESDatasetCiphertextPlaintextPairKey`
- Speck plaintext recovery: `SPECKDatasetCiphertextPlaintext`
- Speck key recovery `SPECKDatasetCiphertextPlaintextPairKey`
- S-AES plaintext recovery: `SimplifiedAESDatasetCiphertextPlaintext`
- S-AES key recovery: `SimplifiedAESDatasetCiphertextPlaintextPairKey`

