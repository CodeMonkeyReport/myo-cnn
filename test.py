import cnn
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
data_folder = 'data-v2-test'
epochs = 500
letters = [chr(i) for i in range(ord('a'), ord('z')+1)]


model = cnn.build_model()
(X, Y) = cnn.get_data(letters, data_folder)

# Latest version of the model is input here, several versions can be found in the models folder.
model.load_weights('model.h5')

res = model.evaluate([X[0].reshape(-1, height, width_emg, 1), X[1].reshape(-1, height, width_acl, 1)], Y, verbose=1)
P = model.predict([X[0].reshape(-1, height, width_emg, 1), X[1].reshape(-1, height, width_acl, 1)], verbose=1)

truth = np.array(Y)
predict = np.array(P)

print("Test set results:")
print("------------------------------------------------")
print(metrics.confusion_matrix(truth.argmax(axis=1), predict.argmax(axis=1)))
print('accuracy:    ', metrics.accuracy_score(truth.argmax(axis=1), predict.argmax(axis=1)))
print('f_1:         ', metrics.f1_score(truth.argmax(axis=1), predict.argmax(axis=1), average=None))
print('recall:      ', metrics.recall_score(truth.argmax(axis=1), predict.argmax(axis=1), average=None))
print('precision:   ', metrics.precision_score(truth.argmax(axis=1), predict.argmax(axis=1), average=None))
print('kappa:       ', metrics.cohen_kappa_score(truth.argmax(axis=1), predict.argmax(axis=1)))