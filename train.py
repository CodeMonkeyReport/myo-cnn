import cnn

width_emg = 8
width_acl = 4
height = 250
data_folder = 'data-v2'
epochs = 500

# Get the model
model = cnn.build_model()

# letters from a to z lower case
letters = [chr(i) for i in range(ord('a'), ord('z')+1)]

# read data
(X, Y) = cnn.get_data(letters, data_folder)

model.summary()
# Data needs to be reshaped to fit model parameters
model.fit([X[0].reshape(-1, height, width_emg, 1), X[1].reshape(-1, height, width_acl, 1)], Y, epochs=epochs)
model.save('model.h5') 