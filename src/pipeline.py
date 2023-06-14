import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from multiprocessing import Pool
from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Add, ReLU
from keras.models import Model
from keras.optimizers import Adam

VALID_ASCII_SET = [np.array([int(bit) for bit in bin(i)[2:].zfill(8)]) for i in range(32, 127)]

#####################
## PARALLELIZATION ##
#####################

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

##########
## DATA ##
##########

def get_dataset_info(train_labels, train_samples, test_labels, test_samples):
    """ Prints shape information about the dataset.
        
        Parameters:
            train_labels (np.array)
            train_samples (np.array)
            test_labels (np.array)
            test_samples (np.array)
    """


    print("===== Training Labels Shape: " + str(np.shape(train_labels)))

    print("===== Label Shape: " + str(np.shape(train_labels[0])))

    print("===== Training Samples Shape: " + str(np.shape(train_samples)))

    print("===== Sample Shape: " + str(np.shape(train_samples[0])))

    print("===== Testing Labels Shape: " + str(np.shape(test_labels)))

    print("===== Testing Samples Shape: " + str(np.shape(test_samples)))

########################
### CREATING A MODEL ###
########################

def residual_block(x, units):
    """ Adds a residual blocks with a specified number of units to the given model.
        
        Parameters:
            x: a Keras model
            units (int): number of units per layer in the residual block
        Returns:
            model with added residual block
    """
    x1 = BatchNormalization()(x)
    x1 = Dense(units=units, activation='relu')(x1)

    x2 = BatchNormalization()(x1)
    x2 = Dense(units=units, activation='relu')(x2)

    out = Add()([x1, x2])
    return out


################################
## MODEL TRAINING AND TESTING ##
################################



def train_model(model: Model, samples, labels, batch_size, epochs):
    """ Trains the given model with the given data and the specified hyperparameters.
        
        Parameters:
            model (Model): the compiled model to train
            samples (np.array)
            labels (np.array)
            batch_size (int)
            epochs (int)
        Returns:
            training history
    """
    history = model.fit(samples, labels, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        shuffle=True,
                        validation_split=0.1)
    return history

def num_correct_bytes(prediction, label):
    """ Returns the number of correct bytes of a prediction according
        to the corresponding label.
        
        Parameters:
            prediction (np.array)
            label (np.array)
        Returns:
            (int): number of correctly predicted bytes
    """
    res = 0
    for i in range(0,len(prediction),8):
        if np.count_nonzero(prediction[i:i+8] != label[i:i+8]) == 0:
            res += 1
    return res

def predict_sample(model: Model, sample):
    """ Predicts a sample's label with the given model
        
        Parameters:
            model (Model)
            sample (np.array): the sample for which to predict the label
        Returns:
            prediction (np.array): the predicted label
    """
    threshold = 0.5

    shape = (1,len(sample))
    output_shape = (len(sample),)

    prediction = model.predict(sample.reshape(shape), batch_size=None)

    bin_prediction = (prediction > threshold).astype(np.uint8)
    bin_prediction = bin_prediction.reshape(output_shape)

    return bin_prediction

def test_model(model: Model, samples, labels, batch_size, ascii_correction: bool):
    """ Tests a model with the given dataset and parameters.
        
        Parameters:
            model (Model): the model we want to test
            samples (np.array): the test samples
            labels (np.array): the test labels
            batch_size (int)
            ascii_correction (bool): whether or not to apply correction to predicted labels
        Returns:
            metrics (dict): a dict with the number of correct bytes, byte accuracy, correct predictions and prediction accuracy
            predictions (np.array): predicted labels by the network
    """

    predictions = model.predict(samples, batch_size=batch_size)

    threshold = 0.5
    metrics = {"correct_predictions": 0, "correct_bytes": 0}
    total_samples = len(predictions)
    total_bytes = total_samples * (len(predictions[0]) // 8)

    data = list(zip(predictions, labels))
    if ascii_correction:
        results = parallelize(data, correct_and_metrics)
    else:
        results = parallelize(data, get_metrics)

    cb = [r[0] for r in results]
    cp = [r[1] for r in results]

    metrics["correct_bytes"] = sum(cb)
    metrics["correct_predictions"] = sum(cp)
    metrics["byte_accuracy"] = metrics["correct_bytes"] / total_bytes
    metrics["accuracy"] = metrics["correct_predictions"] / total_samples

    return metrics, predictions


def test_model_binary(model: Model, samples, labels, batch_size):
    """ Tests a binary classifier model with the given dataset and parameters.
        
        Parameters:
            model (Model): the model we want to test
            samples (np.array): the test samples
            labels (np.array): the test labels
            batch_size (int)
        Returns:
            metrics (dict): a dict with the number correct predictions and prediction accuracy
    """

    predictions = model.predict(samples, batch_size=batch_size)

    metrics = {"correct_predictions": 0}
    total_samples = len(predictions)

    data = list(zip(predictions, labels))

    results = parallelize(data, get_metrics_binary)

    metrics["correct_predictions"] = sum(results)
    metrics["accuracy"] = metrics["correct_predictions"] / total_samples

    return metrics

def test_model_get_correct_predictions(model: Model, samples, labels, batch_size):
    """ Tests a model with the given dataset and parameters.
        
        Parameters:
            model (Model): the model we want to test
            samples (np.array): the test samples
            labels (np.array): the test labels
            
        Returns:
            number of correct predictions (int)
    """
    predictions = model.predict(samples, batch_size=batch_size)

    data = list(zip(predictions, labels))
    correct_predictions = parallelize(data, get_correct_predictions)

    return correct_predictions



def correct_and_metrics(prediction_label_pair):
    """ Given a pair with a prediction and a label, returns the number of 
        correct bytes and the number of correct predictions (1 or 0), after
        ASCII-byte correction.

        Parameters:
            prediction_label_pair (np.array, np.array)
        Returns:
            correct_bytes (int): number of correct bytes
            correct_predictions (int): number of correct predictions (1 or 0)
    """
    prediction, label = prediction_label_pair[0], prediction_label_pair[1]
    threshold = .5
    output_shape = (len(prediction),)
    bin_prediction = (prediction > threshold).astype(np.uint8)
    bin_prediction = bin_prediction.reshape(output_shape)
    bin_prediction = ascii_correction(bin_prediction)

    correct_prediction = 0
    correct_bytes = num_correct_bytes(prediction, label)
    b = len(prediction) // 8
    if correct_bytes == b:
        correct_prediction = 1
    return (correct_bytes, correct_prediction)

def get_metrics(prediction_label_pair):
    """ Given a pair with a prediction and a label, returns the number of 
        correct bytes and the number of correct predictions (1 or 0)

        Parameters:
            prediction_label_pair (np.array, np.array)
        Returns:
            correct_bytes (int): number of correct bytes
            correct_predictions (int): number of correct predictions (1 or 0)
    """
    prediction, label = prediction_label_pair[0], prediction_label_pair[1]
    threshold = .5
    output_shape = (len(prediction),)
    bin_prediction = (prediction > threshold).astype(np.uint8)
    bin_prediction = bin_prediction.reshape(output_shape)

    correct_prediction = 0
    correct_bytes = num_correct_bytes(prediction, label)
    b = len(prediction) // 8
    if correct_bytes == b:
        correct_prediction = 1
    return (correct_bytes, correct_prediction)


def get_correct_predictions(prediction_label_pair):
    """ Given a pair with a prediction and a label, returns the number of 
        correct bytes

        Parameters:
            prediction_label_pair (np.array, np.array)
        Returns:
            correct_bytes (int): number of correct bytes
    """
    prediction, label = prediction_label_pair[0], prediction_label_pair[1]
    threshold = .5
    output_shape = (len(prediction),)
    bin_prediction = (prediction > threshold).astype(np.uint8)
    bin_prediction = bin_prediction.reshape(output_shape)
    bin_prediction = ascii_correction(bin_prediction)

    correct_bytes = []
    for i in range(0,len(prediction),8):
        equals = True
        for j in range(8):
            if prediction[i+j] != label[i+j]:
                equals = False
                break
        if equals:
            correct_bytes.append(prediction[i:i+8])
    return correct_bytes


def get_metrics_binary(prediction_label_pair):
    """ Given a pair with a prediction and a label, returns the number 
        of correct predictions (1 or 0) for binary classifier.

        Parameters:
            prediction_label_pair (np.array, np.array)
        Returns:
            correct_predictions (int): number of correct predictions (1 or 0)
    """
    prediction, label = prediction_label_pair[0], prediction_label_pair[1]
    threshold = .5
    bin_prediction = (prediction > threshold).astype(np.uint8)
    if bin_prediction == label:
        return 1
    else:
        return 0
    

    


def save_model(model, filename):
    """ Saves in the current directory the given model as
        a .h5 file with the given filename.

        Parameters:
            model (Model)
            filename (str)
    """
    model.save(filename + '.h5')

#####################
## POST-PROCESSING ##
#####################

def prediction_to_string(prediction):
    """ Transforms a prediction or label into a string.

        Parameters:
            prediction (np.array): binary array
        Returns:
            (str): corresponding string
    """
    return bytes_to_str(np.packbits(prediction).tobytes()).strip('\x00')

def ascii_correction(prediction):
    """ Corrects the given prediction to contain only valid ASCII bytes.

        Parameters:
            prediction (np.array): binary array
        Returns:
            (np.array): binary arary of valid ASCII bytes
    """
    res = np.array([], np.uint8)
    for i in range(0,len(prediction), 8):
        res = np.concatenate((res, ascii_byte_corrector(prediction[i:i+8])))
    return res

def ascii_byte_corrector(prediction_byte):
    """ Corrects the given byte to a valid ASCII byte, according to Hamming distance.

        Parameters:
            prediction_byte (np.array): byte
        Returns:
            correct_byte (np.array): valid ASCII byte
    """
    min_dist = 9
    min_dist_byte = None
    for b in VALID_ASCII_SET:
        # Hamming distance
        dist = np.count_nonzero(b != prediction_byte)
        if dist == 0:
            return prediction_byte
        if dist < min_dist:
            min_dist = dist
            min_dist_byte = b
    return min_dist_byte

def bytes_to_str(input: bytes) -> str:
    """ Converts bytes to string assuming it is ASCII encoded

        Parameters:
            input (bytes)
        Returns:
            (str)
    """
    return input.decode("ascii")