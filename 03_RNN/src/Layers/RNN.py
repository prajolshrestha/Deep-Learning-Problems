import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


class RNN(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size) -> None:

        super().__init__()
        self.trainable = True
        self._memorize = False #To memorize previous states

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Instances of FC layer
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size) # hidden state
        self.FC_y = FullyConnected(hidden_size, output_size) # output state

        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size)) # For sorting

        # Weights of FC layer
        self.weights_y = None
        self.weights_h = None
        self.weights = self.FC_h.weights

        # Instance of TANH
        self.tan_h = TanH()

        self.bptt = 0
        self.h_t = None # stores current hidden state
        self.prev_h_t = None # stores prev hidden state

        self.batch_size = None
        self.optimizer = None

        self.h_mem = [] # list of storing intermediate hidden states during forward propagation


    def forward(self, input_tensor):

        """
            The forward method takes an input tensor, performs forward propagation through the RNN, and returns the output tensor. 
            Key steps include:

                        -Initializing the hidden state (self.h_t) with zeros.
                        -Iterating through each time step in the input tensor, calculating the new hidden state and output at each step.
                        -Storing the intermediate hidden states in self.h_mem.

                        # concatenating x,ht-1 and 1 to do forwarding to obtain new hidden state ht
                        # for t from 1 to T do:
                        #
                        #       ut = W hh · h t − 1 + W xh · x t + b h --> h t = tanh (x̃ t · W h )
                        # 1.    h t = tanh ( u t )
                        #
                        #       o t = W hy · h t + b y
                        # 2.    ŷ t = σ( o t )
        """

        self.batch_size = input_tensor.shape[0]

        # Initialize hidden states
        if self._memorize: # maintains the use of prev. hidden state
            if self.h_t is None: # if no hidden states, initialize with zeros
                self.h_t = np.zeros((self.batch_size + 1, self.hidden_size)) 
            
            else: # if hidden states exists, use prev. hidden state
                self.h_t[0] = self.prev_h_t

        else:
            self.h_t = np.zeros((self.batch_size + 1, self.hidden_size)) 
        
        # Initialize output state
        y_t = np.zeros((self.batch_size, self.output_size))

        # Forward pass for each time step
        for b in range(self.batch_size):

            # lets manage input and hidden state data by concatinating them
            hidden_ax = self.h_t[b][np.newaxis, :]
            input_ax = input_tensor[b][np.newaxis, :]
            input_new = np.concatenate((hidden_ax, input_ax), axis= 1) # x̃_t

            self.h_mem.append(input_new) # store intermediate hidden state in memory

            #1. Apply TANH
            u_t = self.FC_h.forward(input_new) # apply weights and biases of the hidden layer
            self.h_t[b+1] = TanH().forward(u_t) # h t = tanh (x̃ t · W h ) # new hidden state is calculated
            
            #2. compute output
            y_t[b] = (self.FC_y.forward(self.h_t[b + 1][np.newaxis, :])) # output state

            input_new = np.concatenate((np.expand_dims(self.h_t[b], 0), np.expand_dims(input_tensor[b], 0)), axis = 1)


        self.prev_h_t = self.h_t[-1] # update prev hidden state
        self.input_tensor = input_tensor

        return y_t
    
    def backward(self, error_tensor):

        """
            The backward method computes the gradient of the error with respect to the input and updates the RNN's weights. 
            Key steps include:

                        -Iterating through time steps in reverse order.
                        -Calculating gradients for the output and hidden states.
                        -Backpropagating errors through the FullyConnected layers (self.FC_h and self.FC_y).
                        -Updating the RNN's weights based on the optimization method specified.

                        # 1: for t from 1 to T do:
                        # 2:    Run RNN for one step, computing h_t and y_t
                        # 3:    if t mod k_1 == 0:
                        # 4:        Run BPTT from t down to t-k_2
        """

        self.out_error = np.zeros((self.batch_size, self.input_size))

        self.gradient_weights_y = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_h = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))

        count = 0
        # Gradient Calculation for Tanh Activation:
        grad_tanh = 1 - self.h_t[1::] ** 2 # gradient wrt h_t  

        hidden_error = np.zeros((1, self.hidden_size))

        for b in reversed(range(self.batch_size)):
            
            # Gradient Calculation for Output layer
            yh_error = self.FC_y.backward(error_tensor[b][np.newaxis, :])# gradient wrt output
            
            self.FC_y.input_tensor = np.hstack((self.h_t[b+1], 1))[np.newaxis, :]# update input_tensor

            # Gradient Calculation for Hidden Layer (self.FC_h) and Hidden Error Update:
            grad_yh = hidden_error + yh_error #gradient wrt hidden layer output
            grad_hidden = grad_tanh[b] * grad_yh # gradient wrt hidden state
            xh_error = self.FC_h.backward(grad_hidden) # gradient wrt input to the hidden layer
            hidden_error = xh_error[:, 0:self.hidden_size]

            # Gradient Calculation for Input Features
            x_error = xh_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.out_error[b] = x_error

            con = np.hstack((self.h_t[b], self.input_tensor[b],1)) # concatinated input = hidden state + input_tesor + 1
            self.FC_h.input_tensor = con[np.newaxis, :]#update input_tensor with concatinated input

            #Conditional Update of Weights and Gradients (Backpropagation Through Time - BPTT):
            #  limiting how far back in time the gradients are propagated.
            if count <= self.bptt:
                self.weights_y = self.FC_y.weights
                self.weights_h = self.FC_h.weights

                self.gradient_weights_y = self.FC_y.gradient_weights
                self.gradient_weights_h = self.FC_h.gradient_weights
            count += 1

        # Weight Update Using Optimizer
        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_h)
            
            self.FC_y.weights = self.weights_y
            self.FC_h.weights = self.weights_h


        return self.out_error
    
    ## Properties and Setters: 
    # There are several properties and setters for getting and setting attributes such as 
    # the optimizer, memorization flag, weights, and gradient weights.
    @property
    def gradient_weights(self):
        return  self.gradient_weights_n
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_y.gradient_weights = gradient_weights
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights
        
    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    # initialize the weights and biases of the FullyConnected layers using specified initializer functions.
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)
    

        