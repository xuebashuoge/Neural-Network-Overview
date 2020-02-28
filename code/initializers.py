import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import keras.initializers
import time
import matplotlib.pyplot as plt

def load_data(m=5000, n=100, path='D:/file/vscode/py/data/mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    path = the path of mnist.npz
    if you want to run the code please change the default path
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']

    x_test, y_test = f['x_test'], f['y_test']

    f.close()
    return (x_train, y_train), (x_test, y_test)


# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# use sequential to build cnn model
def CreateModel(initialize_method, input_shape):
    # Another way to build your CNN
    model = Sequential()

    # Conv layer 1 output shape (32, 28, 28)
    model.add(Conv2D(
        input_shape=input_shape,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_first',
        kernel_initializer=initialize_method,
    ))
    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_first',
    ))

    # Conv layer 2 output shape (64, 14, 14)
    model.add(Conv2D(64, 5, strides=1, padding='same', data_format='channels_first',kernel_initializer=initialize_method,))
    model.add(Activation('relu'))

    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initialize_method,))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(10, kernel_initializer=initialize_method,))
    model.add(Activation('softmax'))

    return model

# compile an train the model
def RunModel(X_train, y_train, initialize_method, epoch, batch_size=64):
    # Create model with the required initialization
    model = CreateModel(initialize_method, (1, 28, 28))

    # We add metrics to get more results you want to see
    model.compile(optimizer=Adam(lr=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print('Training ------------')
    #start = time.process_time()
    # Another way to train the model
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,)
    #end = time.process_time()
    #print("CPU executing time: " + str(end-start) + " s")

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, y_test)

    return history, loss, accuracy


Initializations = [keras.initializers.random_normal(mean=0.0, stddev=1.0, seed=1), 
    keras.initializers.glorot_normal(seed=2),
    keras.initializers.he_normal(seed=3),
    keras.initializers.Orthogonal(gain=1,seed=4) ]

# record the result of each model
history = []
loss = []
accuracy = []

for initialization in Initializations:
    hisTemp, lossTemp, accTemp = RunModel(X_train, y_train, initialization, 25)
    
    history.append(hisTemp)
    loss.append(lossTemp)
    accuracy.append(accTemp)

# print the accuracy and loss
print("Using random initialization:")
print("loss: {}".format(loss[0]))
print("accuracy: {}".format(accuracy[0]))
print("{:*^50}".format('*'))

print("Using xavier initialization:")
print("loss: {}".format(loss[1]))
print("accuracy: {}".format(accuracy[1]))
print("{:*^50}".format('*'))

print("Using he initialization:")
print("loss: {}".format(loss[2]))
print("accuracy: {}".format(accuracy[2]))
print("{:*^50}".format('*'))

print("Using orthogonal initialization:")
print("loss: {}".format(loss[3]))
print("accuracy: {}".format(accuracy[3]))
print("{:*^50}".format('*'))

# plot the loss in training
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(history[1].history['loss'], color='blue', label='xaiver initialization')
ax1.plot(history[2].history['loss'], color='red', label='he initialization')
ax1.plot(history[3].history['loss'], color='black', label='orthogonal initialization')
ax1.set_ylabel("Loss")
ax1.set_title('Train History')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper center')

ax2 = ax1.twinx()
ax2.plot(history[0].history['loss'], color='green', label='random initialization')
ax2.set_ylabel("Loss of random initialization")
ax2.legend(loc='upper right')
plt.show()
