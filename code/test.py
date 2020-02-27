import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import time

np.random.seed(1)

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, \
    classes = load_dataset()

# Example of a picture
index = 0
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, [n_x,None], "X")
    Y = tf.placeholder(tf.float32, [n_y,None], "Y")
    
    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   
        
    W1 = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: RELU -> RELU -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.matmul(W1, X) + b1                                       # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                       # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2                                         # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                        # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3                                     # Z3 = np.dot(W3, A2) + b3
    
    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    # use multi cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
        logits=logits, labels=labels))
    
    return cost

def model(X_train, Y_train, X_test, Y_test, epsilon,
          num_epochs, minibatch_size, max = 300000, learning_rate = 0.0001):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    epsilon -- the bound that can be viewed as zero
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    max -- the maximum of iterations
    learning_rate -- learning rate of the optimization
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    start = time.clock()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # count down the iteration
        i = 0

        # a flag for convergence
        flag = 0

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], \
                    feed_dict={X:minibatch_X, Y:minibatch_Y})
                i += 1
                if i % 100 == 0:
                    costs.append(minibatch_cost / num_minibatches)

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost when it reaches a certain number of epochs
            if 20 < minibatch_size < 100:
                if epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            elif minibatch_size >= 100:
                if epoch % 1000 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            elif minibatch_size == 1:
                if epoch % 10 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            
            #stop when it is small enough or can not converge
            if epoch_cost <= epsilon:
                break
            if i >= max:
                flag = 1
                break


        end = time.clock()
        print("CPU executing time = " + str(end - start) + " s" )

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


        if flag == 0:
            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        else:
            print("Failed to converge!")

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test, 0.01, num_epochs=1000000, minibatch_size=64)