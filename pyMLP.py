from scipy import *
from numpy.random import *
import math
seed()


class MLP :

    #The following 3 instance methods are defined in order to calculate the network's output while preventing overflow errors.

    def logistic(self, x):
        if x > 45:
            return 1.0
        elif x < -45:
            return 0.0
        else:
            return 1/(1 + math.exp(-x))

    def tanh(self, x):
        if x > 45:
            return 1.0
        elif x < -45:
            return -1.0
        else:
            return math.tanh(x)

    def identity(self, x):
        return x

    #z and zbar return the network's hidden, and output signals respectively, without preprocessing.

    def zbar(self, x):
        zbar = array(range(len(self.B)), dtype = double)
        zbar = self.gbar(dot(self.B, x))
        return zbar

    def z(self, x, zbar0 = None):
        if zbar0 == None:
            zbar = self.zbar(x)
        else:
            zbar = zbar0
        z = self.g(dot(self.A, zbar))
        return z

    #users should call predict to evaluate their network's output on a single data point.
    #predict applies the stored preprocess to one's data and calculates z the network output.

    def predict(self, x0):
        x = array([concatenate(([1], x0))])
        x = self.preprocess(x)[0]
        return self.z(x)

    #lsem calculates the network's least squares error measure evaluated on a point, without preprocessing.
    #lsem is intended for use within backpropagation only.

    def lsem(self, data):
        x, y = array([i for i in data[0]]), array([i for i in data[1]])
        z = self.z(x)
        e = dot(z - y, z - y)
        return e

    #Users should call lse to gauge the average error of their network on a data set.
    #lse calculates the network's least squares error divided by the number of data points, with preprocessing.

    def lse(self, data0): 
        x = array([concatenate(([1],d[0])) for d in data0])
        x = self.preprocess(x)
        data = array([[x[i], data0[i][1]] for i in range(len(data0))])
        return sum([self.lsem(d) for d in data])/len(data)

    #update applies the stochastic gradient descent update rule to the network's two weight matrices.

    def update(self, data, eta):
        x, y = array([i for i in data[0]]), array([i for i in data[1]])
        zbar = self.zbar(x)
        z = self.z(x, zbar0 = zbar)
        dA, dB = self.delta(x, y, z, zbar)
        maxdA = max(fabs(dA.flatten()))
        maxdB = max(fabs(dB.flatten()))
        if maxdA != 0 and maxdB != 0:
            dA = dA/maxdA
            dB = dB/maxdB
        self.A = self.A - eta*dA
        self.B = self.B - eta*dB

    #cDelta calculates the update term for classifier networks.

    def cDelta(self, x, y, z, zbar):
        k, N, M = len(self.B), len(x) - 1, len(y)
        dA = array([[2*(z[i] - y[i])*z[i]*(1 - z[i])*zbar[j] for j in range(k)] for i in range(M)], dtype = double)
        dB = array([[2*(1 - zbar[i]**2)*x[j]*sum([self.A[l, i]*(z[l] - y[l])*z[l]*(1 - z[l]) for l in range(M)]) for j in range(N + 1)] for i in range(k)], dtype = double)
        return dA, dB

    #rDelta calculates the update term for regression networks.

    def rDelta(self, x, y, z, zbar):
        k, N, M = len(self.B), len(x) - 1, len(y)
        dA = array([[2*(z[i] - y[i])*zbar[j] for j in range(k)] for i in range(M)], dtype = double)
        dB = array([[2*(1 - zbar[i]**2)*x[j]*sum([self.A[l, i]*(z[l] - y[l]) for l in range(M)]) for j in range(N + 1)] for i in range(k)], dtype = double)
        return dA, dB

    #Users should call backpropagation to train their network.
    #backpropagation applies stochastic gradient descent to the network's least squares error evaluated on the training set.
    #backpropagation sets the network's preprocess coefficients, and applies this preprocess to the the training set.
    #preprocess should be set to false if one intends to perform online learning

    def backpropagation(self, data0, epochs = 1, eps = .1, eta = .1, preprocess = True):
        x = array([concatenate(([1],d[0])) for d in data0], dtype = double)
        if preprocess == True:
            self.setPreprocess(x)
            x = self.preprocess(x)
        data = array([[x[i], data0[i,1]] for i in range(len(data0))])
        for i in range(epochs):
            shuffle(data)
            for d in data:
                e = self.lsem(d)
                while e > eps:
                    self.update(d, eta)
                    e = self.lsem(d)

    #preprocess is used to improve network performance.

    def preprocess(self, x0):
        x = array([d for d in x0], dtype = double)
        for d in x:
            d = (d - self.xbar)/self.sigma
        return x

    #setPreprocess calculates and stores a training set's preprocess coefficients.

    def setPreprocess(self, x0):
        x = array([d for d in x0], dtype = object)
        self.xbar, self.sigma = array([0.0 for i in x[0]]), array([0.0 for i in x[0]])
        for i in range(len(self.xbar)):
            self.xbar[i] = (1.0/len(x))*sum([x[i] for d in x])
            self.sigma[i] =((1.0/(len(x)-1))*sum([(x[i] - self.xbar[i])**2  for d in x]))**.5

    #The following method initializes artificial neural network objects.
    #Networks by default include N dimensional input, k hidden units, M dimensional output, and a single bias term at the input layer.

    def __init__(self, N = 1, M = 1, k = 1, role = "r", init = "n"):
        self.role = role
        self.sigma = 1.0
        self.xbar = 0.0
        if init == "n":
            sigmaA, sigmaB = k**.5, (N+1)**.5
            self.A = array([[normal(sigma = sigmaA) for j in range(k)] for i in range(M)], dtype = double)
            self.B = array([[normal(sigma = sigmaB) for j in range(N + 1)] for i in range(k)], dtype = double)
        elif init == "u":
            self.A = array([[uniform(-1, 1) for j in range(k)] for i in range(M)], dtype = double)
            self.B  = array([[uniform(-1, 1) for j in range(N + 1)] for i in range(k)], dtype = double)
        else:
            raise Exception("You have chosen an undefined weight initialization distribution.  The following distributions are included: 'n' (normal), 'u' (uniform) ")
        if self.role == "c":
            self.g = vectorize(self.logistic)
            self.gbar = vectorize(self.tanh)
            self.delta = self.cDelta
        elif self.role == "r":
            self.g = vectorize(self.identity)
            self.gbar = vectorize(self.tanh)
            self.delta = self.rDelta
        else:
            raise Exception("You have chosen an undefined role.  Defined roles include: 'c' (classifier), 'r' (regression).")
