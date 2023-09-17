import numpy as np
from Layers.Base import BaseLayer
from Optimization import Optimizers

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):

        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size, output_size))
        self.bias = np.random.uniform(size=(1, output_size))

        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.temp = []

    
    
    # Add a setter and getter property optimizer which sets and returns the protected member
    # _optimizer for this layer. Properties over a pythonic way of realizing getters and setters.
    # These properties are a way to control access to a class's attributes by providing custom behavior when getting or setting the value of the attribute. 
    
    # By using this combination of the @property decorator and the setter method, 
    # you can create a property named optimizer that allows you to get and set the _optimizer attribute with custom behavior. 
    # This approach is often used to encapsulate the access to class attributes and to introduce additional logic or validation 
    # when reading or writing those attributes.



    # Define a getter using Decorator
    # When you access the optimizer attribute of an instance (e.g., obj.optimizer), this method is automatically called.
    @property
    def optimizer(self): # getter
        return self._optimizer # read _optimizer without directly accessing it
    
    #When you assign a value to the optimizer attribute (e.g., obj.optimizer = new_optimizer), this method will be called automatically.
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def forward(self, input_tensor):

        self.lastIn = input_tensor
        self.lastOut = np.dot(input_tensor, self.weights) + self.bias # y=x.T@w + b

        return self.lastOut


    def backward(self, error_tensor):

        dx = np.dot(error_tensor, self.weights.T)
        dw = np.dot(self.lastIn.T, error_tensor)
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, dw)
            self.bias = self._optimizer.calculate_update(self.bias, error_tensor)
        
        self.gradient_bias = error_tensor
        self.gradient_weights = dw

        return dx
