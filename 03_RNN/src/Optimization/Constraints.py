import numpy as np

class L2_Regularizer():

    def __init__(self, alpha) -> None:
        
        self.alpha = alpha

    def norm(self, weights):

        return self.alpha * np.sum(np.square(weights))

    def calculate_gradient(self, weights):

        return self.alpha * weights

class L1_Regularizer():

    def __init__(self, alpha) -> None:
        
        self.alpha = alpha

    def norm(self, weights):

        return self.alpha * np.sum(np.abs(weights))

    def calculate_gradient(self, weights):

        return self.alpha * np.sign(weights) #sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

    