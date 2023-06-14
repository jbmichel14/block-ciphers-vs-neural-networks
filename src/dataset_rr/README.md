# Datasets for Round-Reduced Ciphers

## Source
The code in this submodule was taken and adapted from the source code for the Paper _An Assessment of Differential-Neural Distinguishers_ by Aron Gohr, Gregor Leander and Patrick Neumann:\
 <https://github.com/differential-neural/An-Assessment-of-Differential-Neural-Distinguishers>

It is useful for the experimentation on Round-Reduced block ciphers, in our case Speck.


## Overview

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Explanation                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| abstract_cipher.py                        | Contains class containing all methods a cipher class should implement the implementation of all ciphers used                                                                                                            |
| speck.py | Implementation of the SPECK cipher submodule                                                                                                             |
| make_train_data.py                     | Methods for the creation of datasets