import numpy as np
from Layers import Base

"""
    Flatten layers reshapes the multidimensional input to  a one dimensional feature vector.
    Connects CNN or Pooling layer with ANN.
"""

class Flatten(Base.BaseLayer):

    def __init__(self):

        super().__init__()

    def forward(self, input_tensor):
        
        self.lastShape = input_tensor.shape  
        batch_size = self.lastShape[0]

        return input_tensor.reshape(batch_size,-1)

        
    
    def backward(self, error_tensor):

        return error_tensor.reshape(self.lastShape)

       
        