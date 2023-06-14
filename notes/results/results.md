# Results


## Plaintext Recovery Attacks

### AES-128 Feed forward

Data generation:
fix cut 16 bytes
2m36s seconds
===== Training Labels Shape: (2234530, 128)
===== Label Shape: (128,)
===== Training Samples Shape: (2234530, 128)
===== Sample Shape: (128,)
===== Testing Labels Shape: (957656, 128)
===== Testing Samples Shape: (957656, 128)




input_shape = np.shape(train_samples[0])

output dimension
dim = len(train_labels[0])

units per hidden layer
units = dim*8
loss_mse = 'mse'
0.1 to 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.01)
optimizer = Adam(learning_rate=lr_schedule)
metrics = ['accuracy', 'binary_accuracy']
epochs = 3
batch_size = 5000

Layer (type)                Output Shape              Param #   

dense_24 (Dense)            (None, 1024)              132096    
                                                                 
 dense_25 (Dense)            (None, 1024)              1049600   
                                                                 
 dense_26 (Dense)            (None, 1024)              1049600   
                                                                 
 dense_27 (Dense)            (None, 1024)              1049600   
                                                                 
 dense_28 (Dense)            (None, 128)               131200  
                                                                 

Total params: 3,412,096
Trainable params: 3,412,096
Non-trainable params: 0


Training:
8m15s

loss stable at 0.2636

Testing:
Correct bytes: 58093
Byte accuracy: 0.00379135357581428
Correct predictions: 0
Accuracy: 0.0

Fixed output ...


### Simplified AES Feed forward

Data generation: 12m0s
===== Training Labels Shape: (17876021, 16)
===== Label Shape: (16,)
===== Training Samples Shape: (17876021, 16)
===== Sample Shape: (16,)
===== Testing Labels Shape: (7661152, 16)
===== Testing Samples Shape: (7661152, 16)

input_shape = np.shape(train_samples[0])

# output dimension
dim = len(train_labels[0])

# units per hidden layer
units = dim*8

loss_scc = 'sparse_categorical_crossentropy'
loss_mse = 'mse'
loss_bce = 'binary_crossentropy'
# 0.1 to 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.01)
optimizer = Adam(learning_rate=0.001)
metrics = ['accuracy', 'binary_accuracy']
epochs = 6
batch_size = 5000

 Layer (type)                Output Shape              Param #   

 dense_6 (Dense)             (None, 128)               2176      
                                                                 
 dense_7 (Dense)             (None, 128)               16512     
                                                                 
 dense_8 (Dense)             (None, 128)               16512     
                                                                 
 dense_9 (Dense)             (None, 128)               16512     
                                                                 
 dense_10 (Dense)            (None, 16)                2064      
                                                                 

Total params: 53,776
Trainable params: 53,776
Non-trainable params: 0

training time: 6m

Correct bytes: 0
Byte accuracy: 0.0
Correct predictions: 0
Accuracy: 0.0

Doesn't always predict the same !!!




Larger network:

exponential decay, 


 Layer (type)                Output Shape              Param #   

 dense_26 (Dense)            (None, 1024)              17408     
                                                                 
 dense_27 (Dense)            (None, 1024)              1049600   
                                                                 
 dense_28 (Dense)            (None, 1024)              1049600   
                                                                 
 dense_29 (Dense)            (None, 16)                16400     
                                                                 
                                                                
Total params: 2,133,008
Trainable params: 2,133,008
Non-trainable params: 0

training time: 41m2s

Correct bytes: 128
Byte accuracy: 8.353835036819528e-06
Correct predictions: 0
Accuracy: 0.0






## Key Recovery Attacks

### Find 1 bit of key

#### AES-128:

Data generation: 1m20s
Strategy: cut every 16 byte (128 bits) --> 1 block
===== Training Labels Shape: (1445705,)
===== Label Shape: ()
===== Training Samples Shape: (1445705, 256)
===== Sample Shape: (256,)
===== Testing Labels Shape: (619588,)
===== Testing Samples Shape: (619588, 256)

Model:
Sequential feed forward
3 Hidden layers
loss_bce = 'binary_crossentropy'
learning_rate = 0.001
optimizer = Adam
metrics = ['binary_accuracy']
epochs = 150
batch_size = 1000


Layer (type)                Output Shape              Param #   

dense_4 (Dense)             (None, 512)               131584    
                                                                 
dense_5 (Dense)             (None, 512)               262656    
                                                                 
dense_6 (Dense)             (None, 512)               262656    
                                                                 
dense_7 (Dense)             (None, 1)                 513       


Total params: 657,409
Trainable params: 657,409
Non-trainable params: 0


Results: 
Training time: 66m
Training loss:
from 0.6936 to 0.4900
Correct predictions: 308967
Accuracy: 0.4986652420640813

Conclusion:
The network did learn something in this case, but nothing useful for unknown data. Probably some overfitting, but a network was able to reduce its loss, which is an interesting result. Requires some deeper research maybe.
Why better loss ("Learning") than others ?

#### S-AES:

Data generation: 10m17s
Strategy: Fix cut every 2 bytes (16 bit) --> 1 block
===== Training Labels Shape: (11565605, 16)
===== Label Shape: (16,)
===== Training Samples Shape: (11565605, 32)
===== Sample Shape: (32,)
===== Testing Labels Shape: (4956688, 16)
===== Testing Samples Shape: (4956688, 32)

Model:
Sequential feed forward
3 Hidden layers
loss_bce = 'binary_crossentropy'
learning_rate = 0.001
optimizer = Adam
metrics = ['binary_accuracy']
epochs = 50
batch_size = 5000

Layer (type)                Output Shape              Param #   

dense_20 (Dense)            (None, 256)               8448      
                                                                 
dense_21 (Dense)            (None, 256)               65792     
                                                                 
dense_22 (Dense)            (None, 256)               65792     
                                                                 
dense_23 (Dense)            (None, 1)                 257       


Total params: 140,289
Trainable params: 140,289
Non-trainable params: 0


Results:
Training until we have some loss convergence
Training time: 49m36s
Training loss:
from 0.6932 to 0.6752

Testing:
Testing time: 
Correct predictions: 333625
Accuracy: 0.5133110853654221

Test distribution: 1 bits: 50.22962913945764 %
                    0 bits: 49.770370860542357 %

Conclusion:
Hard to tell if it learned, better than only predicting 1, better than predicting at random (P=1/2)



#### SPECK32/64:

Data generation: 5m30s
Strategy: Fix cut every 4 byte (32 bit) --> 1 block
===== Training Labels Shape: (5782806,)
===== Label Shape: ()
===== Training Samples Shape: (5782806, 160)
===== Sample Shape: (160,)
===== Testing Labels Shape: (2478345,)
===== Testing Samples Shape: (2478345, 160)

Model:
Sequential feed forward
3 Hidden layers
loss_bce = 'binary_crossentropy'
learning_rate = 0.001
optimizer = Adam
metrics = ['binary_accuracy']
epochs = 50
batch_size = 5000

Layer (type)                Output Shape              Param #  

dense (Dense)               (None, 128)               20608     
                                                                 
dense_1 (Dense)             (None, 128)               16512     
                                                                 
dense_2 (Dense)             (None, 128)               16512     
                                                                 
dense_3 (Dense)             (None, 1)                 129       
                                                                 
Total params: 53,761
Trainable params: 53,761
Non-trainable params: 0

Results:
Training until we have some loss convergence
Training time: 13m58s
Training loss:
from 0.6933 to 0.6881

Testing:
Testing time: 32s
Correct predictions: 1241671
Accuracy: 0.5010081324432232

Test set distribution:
'1' first bit distribution: 0.5017057754267464
'0' first bit distribution: 0.49829422457325356

Conclusion:
About the same as guessing always the same value.
No learning


### Find whole key

#### AES-128

Data generation: 1m20s
Strategy: cut every 16 byte (128 bits) --> 1 block
===== Training Labels Shape: (1445705, 128)
===== Label Shape: (128,)
===== Training Samples Shape: (1445705, 256)
===== Sample Shape: (256,)
===== Testing Labels Shape: (619588, 128)
===== Testing Samples Shape: (619588, 256)

loss_mse = 'mse'
learning_rate = 0.001
optimizer = Adam
metrics = ['binary_accuracy']
epochs = 50
batch_size = 1000

**Feed Forward:**

Layer (type)                Output Shape              Param #   

dense_20 (Dense)            (None, 1024)              263168    
                                                                 
dense_21 (Dense)            (None, 1024)              1049600   
                                                                 
dense_22 (Dense)            (None, 128)               131200    
                                                                 
Total params: 1,443,968
Trainable params: 1,443,968
Non-trainable params: 0

Results: 
Training time: 44m34s
Training loss:
from 0.6936 to 0.4900
Correct predictions: 308967
Accuracy: 0.4986652420640813

Conclusion:

**Residual:**


