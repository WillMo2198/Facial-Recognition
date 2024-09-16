from math import floor, pow
import cv2
from torch import nn

class NN(nn.Module):
    def __init__(self, model_type='nn', chip="cpu"):
        from torch import device
        super().__init__()
        if model_type in ['nn']:    #   TODO: Add more model types later
            self.model_type = model_type
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.device = device(chip)

    '''
    ACTUAL ACTIVATION FUNCTION DECLERATIONS:
    
    @staticmethod
    def relu(x, deriv=False):
        if deriv:
            if x <= 0:
                return 0
            else:
                return x
        return x * (x > 0)

    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
           return sigmoid(x) * (1.0 - sigmoid(x))
        return 1/(1 + np.exp(-x))
    @staticmethod
    def tanh(x, deriv=False):
        if deriv:
            1 - np.exp(np.tanh(x), 2)
        return np.tanh(x)
    '''

    def nonlin(self, nonlin):
        if nonlin == 'relu':
            return self.relu
        elif nonlin == 'sigmoid':
            return self.sigmoid
        elif nonlin == 'softmax':
            return self.softmax
        elif nonlin == 'tanh':
            return self.tanh
        else:
            raise ValueError('Nonlin must be "relu", "sigmoid", "softmax", or "tanh"')

    def build_nn(self, nonlin, dimx=None, dimy=None, out1=None, out2=None, out3=2):
        if dimy is None:    # Input is nxn dimensions
            in_dim = pow(dimx, 2)
        elif dimx is None:    # Input is nxn dimensions
            in_dim = pow(dimy, 2)
        else:    # Input is mxn dimensions
            in_dim = dimx*dimy
        if out2 is None:
            if out1 >= 100:
                out2 = floor(out1/2)
            elif out1 <= 100:
                out2 = out1**2
        stack = nn.Sequential(
            nn.Linear(in_dim, out1),
            self.nonlin(nonlin),
            nn.Linear(out1, out2),
            self.nonlin(nonlin),
            nn.Linear(out1, out2),
            self.nonlin(nonlin),
            nn.Linear(out2, out3),
        )
        return stack

    def forward_nn(self, x, dimx, dimy, nonlin):
        x = self.flatten(x)
        logits = self.build(x, dimx, dimy, nonlin)
        return logits


class Datasets:
    def __init__(self):
        self.faces = cv2.CascadeClassifier('face.xml')
        self.haar_cascade = cv2.CascadeClassifier('face.xml')
