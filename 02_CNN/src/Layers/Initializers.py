import numpy as np

class Constant():

    def __init__(self, constant = 0.1):
        self.constant = constant
        
    def initialize(self, weights_shape, fan_in, fan_out):

        return np.zeros(weights_shape) + self.constant
    
class UniformRandom():

    def __init__(self):
        pass
        
    def initialize(self, weights_shape, fan_in, fan_out):
        
        return np.random.uniform(size=weights_shape)
    
class Xavier():

    def __init__(self):
        pass
        
    def initialize(self, weights_shape, fan_in, fan_out):
        #The scaling factor in Xavier Initialization helps maintain the variance of activations approximately the same across layers. 
        return np.random.randn(*weights_shape) * np.sqrt(2. / (fan_in + fan_out))  

         
    
class He():

    def __init__(self):
        pass
        
    def initialize(self, weights_shape, fan_in, fan_out):
        # Random samples from N(mu=0, sigma = np.sqrt(2/fan_in))
        #Draw random values from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / n_in).
        mu = 0
        sigma = np.sqrt(2. / fan_in)  
        x = sigma * np.random.randn(*weights_shape) + mu

        return x
