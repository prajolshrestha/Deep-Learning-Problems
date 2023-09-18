import numpy as np
from Layers import Base
import copy
from scipy.signal import convolve2d, correlate2d 

"""
    While fully connected layers are theoretically well suited to approximate any function they
    struggle to efficiently classify images due to extensive memory consumption and ovefftting.
    
    Using convolutional layers, these problems can be circumvented by restricting the layer's 
    parameters to local receptive fields.
"""

class Conv(Base.BaseLayer):

    def __init__(self,stride_shape , convolution_shape, num_kernels):

        #super().__init__()
        self.trainable = True

        # Stride shape
        if type(stride_shape) == int:
            stride_shape = (stride_shape,stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        
        # Initialize weights
        self.weights = np.random.uniform(size=(num_kernels,*convolution_shape))

        # Convolution shape
        self.conv2d = (len(convolution_shape) == 3)
        if self.conv2d:
            self.convolution_shape = convolution_shape 
        else:
            self.convolution_shape = (*convolution_shape,1) 
            self.weights = self.weights[:,:,:, np.newaxis]

        # 
        self.num_kernels = num_kernels

        # additional initializations
        self.bias = np.random.uniform(size=(num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.lastShape = None

    def forward(self, input_tensor):
       
        # A. Padded Input Image
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:,:,:, np.newaxis] #expand with an extra dimension
        self.lastShape = input_tensor.shape
        # Initialize padded image to handle boundary cases
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + self.convolution_shape[1] - 1, input_tensor.shape[3] + self.convolution_shape[2] - 1))
        # calculate padding values based on whether is convolution shape even or odd
        p1 = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        p2 = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)

        # check if both the dimension of convolution shape are even
        if self.convolution_shape[1]//2 == 0 and self.convolution_shape[2]//2 == 0:
            padded_image = input_tensor
        else:
            # perform padding
            padded_image[:,:, (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+p1, (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+p2] = input_tensor
            
        input_tensor = padded_image # update image tensor with padded version
        self.padded = padded_image.copy() # store

        ## B. Output Image
        # Calculate the dimensions of the output feature map based on the stride and convolution shape.
        h_cnn = np.ceil((padded_image.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((padded_image.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])

        # initialize output tensor
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        self.output_shape = output_tensor.shape

        # loop through the number of samples
        for n in range(input_tensor.shape[0]):
            # loop through number of filters
            for f in range(self.num_kernels):
                #loop through height of the output
                for i in range(int(h_cnn)):
                    #loop throught width of the output
                    for j in range(int(v_cnn)):
                        # check if within height limits
                        if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= input_tensor.shape[2]) and ((j * self.stride_shape[1]) + self.convolution_shape[2] <= input_tensor.shape[3]):
                            # perform convolution
                            output_tensor[n,f,i,j] = np.sum(input_tensor[n,:, i*self.stride_shape[0] : i*self.stride_shape[0] + self.convolution_shape[1], j*self.stride_shape[1] : j*self.stride_shape[1] + self.convolution_shape[2]] * self.weights[f,:,:,:])
                            output_tensor[n,f,i,j] += self.bias[f]
                        else:
                            output_tensor[n,f,i,j] = 0
                    
        if not self.conv2d:
            output_tensor = output_tensor.squeeze(axis=3) # to solve error in 1D case

         
        return output_tensor
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
    
    def backward(self, error_tensor):

        self.error_T = error_tensor.reshape(self.output_shape)

        if not self.conv2d:
            self.input_tensor = self.input_tensor[:,:,:, np.newaxis]

        # upsample
        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))
        return_tensor = np.zeros(self.input_tensor.shape)
        # For padded input image
        self.de_padded = np.zeros((*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                   self.input_tensor.shape[3] + self.convolution_shape[2]-1))
        
        # Bias and weights
        self.gradient_bias = np.zeros(self.num_kernels)
        self.gradient_weights = np.zeros(self.weights.shape)
        # Padding
        pad_up = int(np.floor(self.convolution_shape[2] / 2)) 
        pad_left = int(np.floor(self.convolution_shape[1] / 2))

        for batch in range(self.up_error_T.shape[0]):
            for kernel in range(self.up_error_T.shape[1]):
                #gradient wrt bias
                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])

                for h in range(self.error_T.shape[2]):
                    for w in range(self.error_T.shape[3]):
                        # fill up with the strided error tensor
                        self.up_error_T[batch, kernel, h* self.stride_shape[0], w* self.stride_shape[1]] = self.error_T[batch, kernel, h, w]

                for ch in range(self.input_tensor.shape[1]): # channel number
                    return_tensor[batch, ch, :] += convolve2d(self.up_error_T[batch, kernel, :], self.weights[kernel, ch, :], 'same') # zero padding

            # Delete the padding
            for n in range(self.input_tensor.shape[1]):
                for h in range(self.de_padded.shape[2]):
                    for w in range(self.de_padded.shape[3]):
                        if (h > pad_left - 1) and (h < self.input_tensor.shape[2] + pad_left):
                            if (w > pad_up -1) and (w < self.input_tensor.shape[3] + pad_up):
                                self.de_padded[batch, n, h, w] = self.input_tensor[batch, n ,h - pad_left, w - pad_up]
            
            for kernel in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):
                    #convolution of the error tensor with the padded input tensor
                    self.gradient_weights[kernel, c, :] += correlate2d(self.de_padded[batch, c, :], self.up_error_T[batch, kernel, :], 'valid') #valid padding

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if not self.conv2d:
            return_tensor = return_tensor.squeeze(axis=3) 


        return return_tensor
    
    def initialize(self, weights_initializer, bias_initializer):

        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
