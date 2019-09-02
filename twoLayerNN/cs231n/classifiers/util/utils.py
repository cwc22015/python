import numpy as np

class forward_backward_propogation(object):

    def __init__(self, W1, b1, W2, b2, X=0):
        """
        Only for two layer neural net
        """
        # get dimensions of nodes at each layer

        self.X = X                                                    # dim (N0, D)
        self.W1 = W1                                                  # dim (D, N1)
        self.b1 = b1                                                  # dim (1, N1)
        self.z1 = 0.0                                                 # dim (N0, N1)
        self.a1 = 0.0                                                 # dim (N0, N1)
        self.W2 = W2                                                  # dim (N1, N2)
        self.b2 = b2                                                  # dim (1, N2)
        self.z2 = 0.0                                                 # dim (N0, N2)
        self.a2 = 0.0                                                 # dim (N0, N2)
        self.dz1 = 0.0                                                # dim (N0, N1)
        self.dW1 = 0.0                                                # dim (D, N1)
        self.db1 = 0.0                                                # dim (1, N1)
        self.da1 = 0.0                                                # dim (N0, N2)
        self.dz2 = 0.0                                                # dim (N0, N2)
        self.dW2 = 0.0                                                # dim (N1, N2)
        self.db2 = 0.0                                                # dim (1, N2)
        self.da2 = 1.0
        self.loss = 0.0

    def start(self, X):
        self.X = X

        N0 = self.X.shape[0]
        D = self.X.shape[1]
        N1 = self.W1.shape[1]
        N2 = self.W2.shape[1]

        self.z1 = np.zeros((N0, N1))                                  # dim (N0, N1)
        self.a1 = np.zeros((N0, N1))                                  # dim (N0, N1)
        self.z2 = np.zeros((N0, N2))                                  # dim (N0, N2)
        self.a2 = np.zeros((N0, N2))                                  # dim (N0, N2)
        self.dz1 = np.zeros((N0, N1))                                 # dim (N0, N1)
        self.dW1 = np.zeros((D, N1))                                  # dim (D, N1)
        self.db1 = np.zeros((1, N1))                                  # dim (1, N1) 
        self.da1 = np.zeros((N0, N1))                                 # dim (N0, N2)
        self.dz2 = np.zeros((N0, N2))                                 # dim (N0, N2)
        self.dW2 = np.zeros((N1, N2))                                 # dim (N1, N2)
        self.db2 = np.zeros((1, N2))                                  # dim (1, N2)
        self.da2 = 1.0
        self.loss = 1.0

    def refresh():
        """
        reset list values to 0.0
        """
        self.z1 *= 0.0
        self.a1 *= 0.0
        self.z2 *= 0.0
        self.a2 *= 0.0
        self.dz1 *= 0.0
        self.dW1 *= 0.0
        self.db1 *= 0.0
        self.da1 *= 0.0
        self.dz2 *= 0.0
        self.dW2 *= 0.0
        self.db2 *= 0.0
        self.da2 *= 0.0
        self.loss *= 0.0

    def forward_scores(self, X, W, b):
        """
        X:  Input features; has shape (N, D)
        W: First layer weights; has shape (D, H)
        b: First layer biases; has shape (1, H)
        z: output matrix; has shape (N, H)
        """
        z = X.dot(W) + b.reshape(1, -1)
        return z

    def backward_scores(self, X, dz):
        dW = X.T.dot(dz)
        db = dz
        return (dW, db)

    def forward_activation_softmax(self, z, y=None):
        """
        z:  scores matrix; has shape (N, H)
        y:  actual labels; has shape (1, H)
        """
        z -= np.max(z, axis=1)
        f = np.exp(z)
        # summing over all hidden nodes H in the layer
        sum_f = np.sum(f, axis=1).reshape(-1,1)
        # activation function - softmax
        a = np.zeros((z.shape[0], z.shape[1]))
        return np.exp(z) / sum_f

    def backward_activation_softmax(self, da, a):
        mat_id = np.identity(n=(a.shape[1]))
        grad_dadz = a*mat_id - a.T.dot(a)
        return grad_dadz.dot(da)

    def forward_activation_ReLU(self, z):
        return np.maximum(0, z)

    def backward_activation_ReLU(self, da, a):
        result = np.zeros((a.shape[0], a.shape[1]))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i,j] > 0.0:
                    result[i,j] = 1.0
                else:
                    result[i,j] = 0.0
    
        return (result * da)

    def forward(self, X):
        # reset
        self.refresh()
        # score at the 1st layer
        self.z1 = self.forward_scores(X, self.W1, self.b1)
        # activation value at the 1st layer
        self.a1 = self.forward_activation_ReLU(z1)
        # score at the 2nd layer
        self.z2 = self.forward_scores(a1, self.W2, self.b2)
        return self.z2

    def backward(self, loss):

        return 1.0

