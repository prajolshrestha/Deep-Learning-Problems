import numpy as np

class CrossEntropyLoss(object):

    def __init__(self):
        pass
    
    # __call__ method can make code more concise and readable because it allows you to treat instances of the class like functions
    # result = loss(argument1, argument2)
    # result = loss.forward(argument1, argument2) # Both are equivalent
    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, prediction_tensor, label_tensor):
        self.lastIn = prediction_tensor
        y_hat = prediction_tensor
        y = label_tensor
        loss = -np.sum(y * np.log(y_hat + np.finfo(float).eps))
        return loss
    
    def backward(self, label_tensor):
        return -(label_tensor / (self.lastIn + np.finfo(float).eps))