import matplotlib.pyplot as graph
import numpy
import random

# Setup
# Generating random values
x = [random.uniform(0, 1) for i in range(100)]


# Generating test data set
def getTestData(training, data, test):
    idx = 0
    for x in data:
        if x not in training:
            test[idx] = x
            idx += 1
    return test


# Generating the training and test data sets
xTraining = random.sample(x, 70)
yTraining = [-.5 + .6 * xTraining[i] + .01 * (xTraining[i] ** 3) + 2 * (xTraining[i] ** 6)
             for i in range(70)]
array30 = [0 for i in range(30)]
xtest = getTestData(xTraining, x, array30)
ytest = [-.5 + .6 * xtest[i] + .01 * (xtest[i] ** 3)
         + 2 * (xtest[i] ** 6) for i in range(30)]
xTraining.sort()
xtest.sort()

# Generating noise
noise = numpy.random.normal(0, 1, 100)

# applying noise to data set
ynoise = [-.5 + .6 * x[i] + .01 * (x[i] ** 3) + 2 * (x[i] ** 6) + noise[i] for i in range(100)]

# Part a
# plotting the data
graph.scatter(x, ynoise, label="Data with Noise")

# Sorting the x data to plot it
x2 = x
x2.sort()
xsorted = x2

# Calculating true function without noise
y = [-.5 + .6 * xsorted[i] + .01 * (xsorted[i] ** 3) + 2 * (xsorted[i] ** 6) for i in range(100)]
graph.plot(x, y, label="True Function")


# Part b
# Setting up the 'X' matrix
def setupx(xdata, num):
    x = [[0, 0] for k in range(num)]
    for i in range(num):
        for j in range(2):
            if (j == 0):
                x[i][j] = xdata[i] ** 3
            else:
                x[i][j] = xdata[i] ** 6
    return x


# Setting up the 'Y' matrix
def setupy(ydata, num):
    y = [[ydata[n]] for n in range(num)]
    return y


# Calculating the missing values
def calcTheta(x, y):
    xt = numpy.transpose(x)
    xtx = numpy.matmul(xt, x)
    inverse = numpy.linalg.inv(xtx)
    xty = numpy.matmul(xt, y)
    return numpy.matmul(inverse, xty)


# Calculating model using values found
def equation(theta, data, num):
    return [(-.5 + .6 * data[k] + theta[0][0] * (data[k] ** 3)
             + theta[1][0] * (data[k] ** 6))
            for k in range(num)]


# Using the training data to plot the test data
theta = calcTheta(setupx(xTraining, 70), setupy(yTraining, 70))
print(theta)
graph.plot(xtest, equation(theta, xtest, 30), label='Least Squares Method')
graph.legend()
graph.show()

# Part c
# Setting up the original 'theta', 'p', and 'b' matrices
t = [[0], [0]]
p = [[1, 0], [0, 1]]
b = [[0], [0]]


# Preform either recursive or forgetting least squares
def recurse_ls(idx, val):
    global b, t

    # Transforming the 'p' array
    x_arr = [[xTraining[idx] ** 3], [xTraining[idx] ** 6]]
    top = numpy.matmul(numpy.matmul(numpy.matmul(p, x_arr), numpy.transpose(x_arr)), p)
    bottom = val + numpy.matmul(numpy.matmul(numpy.transpose(x_arr), p), x_arr)[0]

    # Calculating new values for 'p' array
    for i in range(2):
        for j in range(2):
            top[i][j] = top[i][j] * bottom
            p[i][j] = p[i][j] - top[i][j]
            p[i][j] = p[i][j] / val

    # Calculating new values for 'b' array & combing 'b' and 'p'
    xy = [[(xTraining[idx] ** 3) * yTraining[idx]], [(xTraining[idx] ** 6) * yTraining[idx]]]
    b = [[b[0][0] + xy[0][0]], [b[1][0] + xy[1][0]]]
    t = numpy.matmul(p, b)
    return t


# Resetting the values for forgetting least squares
def reset():
    global t, p, b
    t = [[0], [0]]
    p = [[1, 0], [0, 1]]
    b = [[0], [0]]


# Perform recursive and forgetting least squares over the entire training data set
def get_theta(val):
    for i in range(70):
        new_t = recurse_ls(i, val)
    return new_t


# Getting necessary information from the data set
r_theta = get_theta(1)
r_data = equation(r_theta, xtest, 30)


# Calculate MSE for certain data set
def mse(y, yhat, num):
    sum = 0;
    for i in range(num):
        sum += (y[i] - yhat[i]) ** 2
    return sum / num


# Calculating MSE for the least squares and recursive least squares methods
mse1 = mse(ytest, equation(theta, xtest, 30), 30)
mse2 = mse(ytest, r_data, 30)

print(mse1)
print(mse2)

# Part d
# Resetting the data for forgetting least squares
reset()

# Calculating MSE for the forgetting least squares method
f_theta = get_theta(.85)
f_data = equation(f_theta, xtest, 30)
fmse = mse(ytest, f_data, 30)

print(fmse)
