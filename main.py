"""
    This is my first deep neural network from scratch
"""

# IMPORTS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# MAIN PROGRAMM
data = pd.read_csv('./train.csv/train.csv')
# print(data.head())  # here we should see a table with 783 columns, and 5 rows

# we want to work with numpy arrays
data = np.array(data)
# print("data:", data)
# now we split our data into 'dev' and 'training' sets
m, n = data.shape  # 'm' stands for the rows (42000), 'n' stands for the columns (785)
print(f"\n\n========== m (rows): {m},  n (columns): {n} ==========")
np.random.shuffle(data)  # This only shuffles the array along the first axis. The order of sub-arrays is changed but their contents remains the same
# print("data:", data)
# print("data:", data[m - 1])

# print("data before transposing:", data[0:1000])
# print("data_dev after transposing:", data[0:1000].T)
data_dev = data[0:1000].T  # 'T' is for transpose the vector formed with each row, to work with columns instead
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255  # we normalize the values of the pixels (0, 255) into -> (0, 1)
"""
    The main reason we do transposing is because, we rather to work with every pixel in the same position for every image
    that we will use to train the model, so, we need to work first with the first pixel of every image (row), and then
    with the second one of  every image (row), and so on...

    data before transposing: [
        [9 0 0 ... 0 0 0]
        [6 0 0 ... 0 0 0]
        [3 0 0 ... 0 0 0]
        ...
        [5 0 0 ... 0 0 0]
        [1 0 0 ... 0 0 0]
        [2 0 0 ... 0 0 0]
    
    ]
    data_dev: [
        [9 6 3 ... 5 1 2]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        ...
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
    ]
"""
data_train = data[1000:m].T  # from the row 1000 to the row 42,000 for this case
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255
# print("y_train:\n", y_train)
# print("x_train:\n", x_train)
# print("x_train[:, 0]:\n", x_train[:, 0])
# print("x_train[:, 0].shape:\n", x_train[:, 0].shape)
"""
    The colon ":" in the first position indicates the selection of all rows in the array.
    The 0 in the second position indicates the selection of the first column of the array.

    Together, [:, 0] means "select all rows from the first column."
    This is a way to extract the entire first column of a 2D array (matrix).
    
    Here's a more complete example for clarity:
    
    # Create a sample 2D NumPy array
    my_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Use the syntax to select all rows from the first column
    first_column = my_array[:, 0]

    print(first_column)

    In this example, first_column would output:
    [1 4 7] 
"""

# ========== FUNCTIONS =============================================================================
def init_params():
    # W1 is for the first layer of weights, the first one will be 784 neurons, that will be connected to 10 neurons in the second layer, each connection (weight) with a random value between 0 and 1
    W1 = np.random.rand(10, 784) - 0.5  # np.random.rand makes random values between 0 and 1. the 2 arguments (10, 784) describes the shape of the resulting array
    B1 = np.random.rand(10, 1) - 0.5  # first argument is for the number of rows, while second is for the number of columns
    # B1 is for the first vector or array of "bias" values, corresponding to the first layer
    # W2 is for the second layer of weights associated with one of the neurons of the first layer to the second layer, since the first layer is now of 10 neurons, this second array of weights should be of 10 rows by 10 columns 
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5  # we substract -5 to get values between -0.5 and 0.5, this for practical uses with numpy and np.exp() method
    # print("W1:\n", W1)
    # print("W1.shape:\n", W1.shape)
    """
        Example: np.random.rand(3,2)
        array(
            [
                [ 0.14022471,  0.96360618],  #random
                [ 0.37601032,  0.25528411],  #random
                [ 0.49313049,  0.94909878]   #random
            ]
        ) 
    """
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)  # this goes through every element in Z, and if the element is greater than 0, returns Z, if is not, returns 0


def softmax(Z):
    """
        np.exp(Z) it just applies "e" to the "Z" to every single value of Z 

        sum is going to sum up through each value of Z substituted in e^Z[j]

        the point of this is to get values based on mutually exclusive probabilities,
        this is, getting values between 0 and 1, that summed up we get the value of 1 
    """
    return np.exp(Z) / sum(np.exp(Z))
    

def forward_propagation(W1, B1, W2, B2, X):
    # print("W1.dot(X):\n", W1.dot(X))
    # print("W1.dot(X).shape:\n", W1.dot(X).shape)
    
    # print("B1:\n", B1)
    # print("B1.shape:\n", B1.shape)

    Z1 = W1.dot(X) + B1  # Z1 = Sum(Wij * Xji) + Bk; 
    """
        Sum(Wni * Xjn) throws as output:

            A matrix of (10 [rows], 41000 [columns])
            And every each of the elements of this new matrix, 
            has been added with the Bn element corresponding,
            since B1 is a matrix of (10 [rows], 1 [column]), this
            matrix has to be "streched" through the resulting matrix of
            Sum(Wni * Xjn) of (10, 41000),
            so we can make the addition of matrix:
            W1.dot(X) + B1
            resulting in an array with shape of
            (10, 41000)

            in the Sum() function,
            we start i = 0, j = 0,
            until both reach "n" which is 784 for this case, 
            next proceed to add the "bias", 
            and this will not change its value of "k" until
            we had added this bias to every column resulting
            of the dot product between W1 and X1, by every row,
            this is the "stretching" process mentioned before
    """
    # print("Z1:\n", Z1)
    # print("Z1.shape:\n", Z1.shape)
    A1 = ReLU(Z1)  # here use an activation function to "squish" the Z1 vector values
    # print("A1:\n", A1)
    # print("A1.shape:\n", A1.shape)
    Z2 = W2.dot(A1) + B2
    # print("Z2:\n", Z2)
    # print("Z2.shape:\n", Z2.shape)
    A2 = softmax(Z2)
    # print("A2:\n", A2)
    # print("A2.shape:\n", A2.shape)
    return Z1, A1, Z2, A2


def one_hot(Y):
    """
        it returns a vector of 0's and
        with one "1" in the position it should be the correct answer
        of the NN

        "Y.size" is the 41000 "columns" of Y,
        resulting in an array of 41000 columns, by 10 rows,
        since "Y.max()" will return 9, since,
        max() method finds the biggest value in Y, which is has to be 9,
        since we are labeling from 0 to 9 (numeric values), then we add 1,
        therefore, we have "10" rows.
    """
    one_hot_y = np.zeros( (Y.size, Y.max() + 1) )  # vector of 0's with shape of ( 41000, 10 
    one_hot_y[np.arange(Y.size), Y] = 1  # replace "1" when the position of that component its the same as the label
    # print("one_hot_y:\n", one_hot_y)
    # print("one_hot_y.shape:\n", one_hot_y.shape)
    """
        since "one_hot_y" its an array of (41000 [rows], 10 [columns])
        made full of 0s,
        at the moment we do:
        
        one_hot_y[np.arange(Y.size), Y] = 1 
        
        what we are doing with 
        "np.arange(Y.size)" 
        is to generate an array of (41000,) with values
        from 0 to 40,999
        this is because
        Y.size returns the size of the array Y, which, 
        is 41000. This is because Y is an array with 41,000 elements, 
        each representing a class label for a sample.
        
        At the moment we pass the first argument to:
        "one_hot_y[first_argument, second_argument]"
        we are saying that we will go through the rows 0 to 40,999
        of the one_hot_y array with shape of (41000, 10)
        and for every row of it, we will go to the position pointed by
        the value of Y of the curren value,
        Y has a shape of (41000,) too, but its values are between:
        0 and 9,
        so, Y in 
        
        one_hot_y[np.arange(Y.size), Y] = 1
        
        indicates at what index of the sub-array of every row of 
        one_hot_y
        will have a value of "1"
        so, this will make a bidimentional matrix,
        where each row (41000 in total) will store a vector
        of 10 elements, all equals to 0 but 
        the one that was altered to 1 
        pointing at what label of class corresponds 
        every row of the one_hot_y matrix, that in turn,
        it corresponds to every and each one of the samples
        used to train the NN.

        Yet, we need to transpose it...
    """
    one_hot_y = one_hot_y.T
    # print("one_hot_y transposed:", one_hot_y)
    # print("one_hot_y_transposed.shape:\n", one_hot_y.shape)
    return one_hot_y


def deriv_ReLU(Z):
    return Z > 0


def back_propagation(Z1, A1, A2, W2, X, Y):
    """
        except for the one_hot() function,
        everything here still needs math demostration, but
        will be set for later tho...

        just to know we need to know how to derivate and
        the implications of working with linear algebra
    """
    one_hot_Y = one_hot(Y)  # it returns a binary matrix, each column (10 in total) made of 0's but one element equal to "1" in the position it should be the correct answer of the NN
    dZ2 = A2 - one_hot_Y  # dont know how this outcame with that formula
    # dZ2 corresponds to the "loss" of the function, the "error" of the outcame with that formula from the NN
    dW2 = ( 1 / m ) * dZ2.dot(A1.T)  # dont know how this outcame with that formula
    dB2 = ( 1 / m ) * np.sum(dZ2)  # dont know how this outcame with that formula
    # dW2M and dB2 corresponds to the new weights and bias to be updated later with an extra step within
    # dZ1 corresponds to the "error" but this time among the 1st and 2nd layer of neurons
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)  # dont know how this outcame with that formula
    dW1 = ( 1 / m ) * dZ1.dot(X.T)  # dont know how this outcame with that formula
    dB1 = ( 1 / m ) * np.sum(dZ1)  # dont know how this outcame with that formula
    return dW1, dB1, dW2, dB2


def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    """
        this still needs math demostration, or at least,
        a solid reason of why this update process goes
        like this...

        considering and "alpha" parameter set arbitrary by you,
        and not by the NN,
        and why it is substracted from the matrix Wn and Bn,
        and why multiplied by the "derivates" of Wn and Bn
    """
    W1 = W1 - (alpha * dW1)
    B1 = B1 - (alpha * dB1)
    W2 = W2 - (alpha * dW2)
    B2 = B2 - (alpha * dB2)
    return W1, B1, W2, B2


# ================================================================================================
def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    print("Y (labels):", Y)
    # print("Y.shape:", Y.shape)
    print("X (pixels):", X)
    # print("X.shape:", X.shape)
    print("iterations:", iterations)
    print("alpha:", alpha)
    print("============================\n\n")
    W1, B1, W2, B2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B2, W2, B2, X)
        dW1, dB1, dW2, dB2 = back_propagation(Z1, A1, A2, W2, X, Y)
        W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if i % 20 == 0:  # every 10th iteration, will print the iteration number and the accuracy of the NN at that iteration
            print("Iteración: ", i)
            print("Precision: ", get_accuracy(get_predictions(A2), Y))
    
    return W1, B1, W2, B2


# ================================================================================================
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]  # this is a matrix of (784, 1), made up of every and each pixel of one sample image
    # print("X:", x_train[:, index, None])
    # print("X.shape:", x_train[:, index, None].shape)
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# ================================================================================================
# RUN THE MAIN PROGRAM
W1, B1, W2, B2 = gradient_descent(x_train, y_train, 261, 0.8)
test_prediction(0, W1, B1, W2, B2)
test_prediction(1, W1, B1, W2, B2)
test_prediction(2, W1, B1, W2, B2)
test_prediction(3, W1, B1, W2, B2)