import keras
from keras import layers
from keras import backend
import fnmatch
import os
import numpy as np
import sklearn.metrics as metrics

width_emg = 8
width_acl = 4
height = 250
data_folder = 'data-v2'
epochs = 1000

optimizer = keras.optimizers.RMSprop( lr=0.001, clipvalue=1.0)

# letters from a to z lower case
letters = [chr(i) for i in range(ord('a'), ord('z')+1)]

def get_positive_label(letter):
    output_label = [0] * len(letters)
    output_label[letters.index(letter)] = 1
    return output_label

def get_data(letters, data_folder):

    X_1 = []
    X_2 = []
    Y = []
    for i, letter in enumerate(letters):
        for file_name in fnmatch.filter(os.listdir(data_folder), "[0-9]-{0}*.csv".format(letter)):
            data = np.loadtxt(data_folder + '/' + file_name, delimiter=',')
            X_1.append(data[:,0:8])
            X_2.append(data[:,8:])
            Y.append(get_positive_label(letter))
    return ([np.array(X_1), np.array(X_2)], np.array(Y))


def build_model():
    input_layer_one = keras.Input(shape=(height, width_emg, 1))
    m = layers.Conv2D( 32, (5, 5), activation=backend.sigmoid )(input_layer_one)
    # 124, 5
    m = layers.Conv2D( 32, (3, 3), activation=backend.sigmoid )(m)
    m = layers.AveragePooling2D()(m)

    input_layer_two = keras.Input(shape=(height, width_acl, 1))
    m_2 = layers.Conv2D(32, (7, 3))(input_layer_two)
    m_2 = layers.MaxPool2D()(m_2)

    m = layers.merge.Concatenate()([m, m_2])
    m = layers.Reshape((122, 64))(m)

    m = layers.LSTM(16)(m)

    m = layers.Dense(52, activation=backend.sigmoid)(m)
    m = layers.Dropout(0.5)(m)
    m = layers.Dense(26, activation=backend.sigmoid)(m)

    model = keras.models.Model( inputs=[input_layer_one, input_layer_two], outputs=m )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model
