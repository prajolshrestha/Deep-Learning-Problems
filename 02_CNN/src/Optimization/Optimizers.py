# import numpy as np
# """
# A.Simple Optimizer:
# 1.Stochastic Gradient descent (SGD)
#     Key Components:
#             Stochastic: The term "stochastic" refers to the fact that instead of computing the gradient of the loss function 
#                         using the entire dataset (as in standard gradient descent), SGD uses a random subset of the data at each iteration.    

#     Steps: 1. Initialization (initialize Weights and biases)
#            2. Iterative Optimization
#                 2.1 Random mini-batch (Random samples from training dataset)
#                 2.2 Forward Pass (Compute Activations) ie. models predictions
#                 2.3 Compute Loss (Compute error between actual target and predicted values)
#                 2.4 Backward Pass (Compute Gradients) ie. gradient of loss wrt model's parameters ==> represents direction and mag. of change needed to minimize the loss
#                                ie, (aims to minimize a loss function by iteratively updating the model's parameters (weights and biases).)                                   
#                 2.5 Update Parameters (update weights and biases)
#                                        θ = θ - α * gradient
#                 2.6 Repeat

#     Benifits: -Computationally Efficient (works only on a small subset of data at each iteration)
#               -Regularization (due to stochastic nature which introduces a form of noise during training, helping prevent overfitting)
#               -Quick Convergence (converges quickly even when loss surface is noisy or non-convex)

#     Challenges: -Noisy updates(minibatch introduces noise in the parameter updates, which can lead to fluctuating convergence)
#                 -Learning rate selection

#     Varients of SGD: To improve convergence and stability (ADAM, SGD with momentum)
# """
# class Sgd():

#     def __init__(self, learning_rate: float):
#         self.learning_rate = learning_rate

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         """
#             Update Parameters: Update the model's parameters using the computed gradients. 
#                                This step involves adjusting the weights and biases in the opposite direction of the gradients 
#                                to minimize the loss.
#         """

#         weight_tensor = weight_tensor - self.learning_rate*gradient_tensor

#         return weight_tensor
    
# ####################################################################################################################################
# """
# B.More advanced Optimizers:
# 1. SGD with Momentum:
#     SGD algorithm + a momentum term that helps accelerate convergence and reduces oscillations during training.
#     (Aims to minimize a loss function by iteratively updating the model's parameters (weights and biases).)
    
#     Key Components:
#                 Momentum: Running average of past gradients. 
#                         It helps to smooth out the updates and accelerates convergence,
#                         (particularly where the loss surface has high curvature or noisy gradients)

#     Steps: 1. Initialize Model Parameters (Weights and biases)
#            2. Initialize Momentum(ß) (usually between 0 and 1) ==> higher value means stronger momentum effect
#            3. Iterative Optimization until convergence
#                 2.1 Random mini-batch 
#                 2.2 Forward Pass (Compute Activations)
#                 2.3 Compute Loss (Compute error between actual target and predicted values)
#                 2.4 Backward Pass (Compute Gradients) ie. gradient of loss wrt model's parameters 
#                 2.5 Update momentum
#                             - Accumulate a fraction(ß) of current gradient and add it to the previous momentum  
                            
#                             velocity = β * velocity + α * gradient   #velocity:Accumulated gradient information
               
#                 2.6 Update Parameters using momentum-influnced gradient descent. (update weights and biases)
#                              θ = θ - velocity
#                 2.7 Repeat

#     Benifits:-Faster convergence
#              -Reduced Oscillations
#              -Escape local minima

# """
# class SgdWithMomentum():
    
#     def __init__(self, learning_rate, momentum_rate):
        
#         # Hyperparameter
#         self.learning_rate = learning_rate
#         self.momentum_rate = momentum_rate
#         self.velocity = 0.

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         '''
#              "velocity" represents the accumulation of gradient information from previous iterations., 
#             and the model's parameters are updated based on this velocity.

#             velocity term is crucial for controlling the step size and direction of parameter updates 
#         '''
#         # compute velocity and update it
#         velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor
#         self.velocity = velocity
#         # Update parameters
#         weight_tensor = weight_tensor - velocity
        
        
#         return weight_tensor
# '''
# 2. ADAM:(Adaptive Moment Estimation) ==> RMSprop + momentum

#         Key Components: a.Exponential Moving Averages: ADAM maintains two exponentially moving avegrages of past gradient information.
#                                     - First moment(mean)
#                                     - Second Moment(uncentered variance)
                        
#                         b.Bias Correction: To mitigate bias towards zero in the early stages of training.

#         Steps:  1. Initialize Model Parameters (Weights and biases)
#                 2. Initialize First Moment(mean) and Second Moment(uncenterd variance) moving average for each parameter
#                 3. Iterative Optimization until convergence
#                         2.1 Random mini-batch 
#                         2.2 Forward Pass (Compute Activations)
#                         2.3 Compute Loss (Compute error between actual target and predicted values)
#                         2.4 Backward Pass (Compute Gradients) ie. gradient of loss wrt model's parameters 
#                         2.5 Update First moment and Second Moment moving averages.
#                                         m_t = β1 * m_(t-1) + (1 - β1) * ∇θL(θ)      # First Moment
#                                         v_t = β2 * v_(t-1) + (1 - β2) * (∇θL(θ))^2  # Second Moment
#                                                 # β1 and β2 are hyperparameters controlling the exponential decay rates of the moving averages (typically close to 1).
#                                                 # t = iteration
#                         2.6 Perform Bias Correction on the moving average.
#                                         m_t_hat = m_t / (1 - β1^t)  # Bias correction for the first moment 
#                                         v_t_hat = v_t / (1 - β2^t)  # Bias correction for the second moment
#                         2.7 Compute the effecting learning rate for each parameter.  
#                                         θ = θ - α * m_t_hat / (sqrt(v_t_hat) + ε) 
#                         2.5 Update Parameters based on the moving avegrages and the effective learning rates. (update weights and biases)
#                         2.6 Repeat
#         Benifits: - Adaptive Learning rates
#                   - Robustness
#                   - Efficiency

#         Challanges: In addition to the choice of β1 and β2, hyperparameters like the learning rate (α) and ε should be tuned for specific problems
        

# '''
# class Adam():

#     def __init__(self, learning_rate, mu, rho):

#         self.learning_rate = learning_rate
#         self.mu = mu
#         self.rho = rho
#         self.first_moment = 0.
#         self.second_moment = 0.
#         self.t = 1

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         # Update moment
#         self.first_moment = self.mu * self.first_moment + ((1 - self.mu) * gradient_tensor)
#         self.second_moment = self.rho * self.second_moment + ((1 - self.rho) * np.power(gradient_tensor,2))
#         # Bias correction
#         first_moment_hat = self.first_moment / (1 - np.power(self.mu,self.t))
#         second_moment_hat = self.second_moment / (1 - np.power(self.rho,self.t))
#         # Next iteration
#         self.t +=1
#         # Adapt learning rate
#         # self.learning_rate = self.learning_rate * (np.sqrt(1 - np.power(self.rho,self.t)) / (1 - np.power(self.mu,self.t)))
#         # Update parameters
#         weight_tensor = weight_tensor - self.learning_rate * (first_moment_hat / (np.sqrt(second_moment_hat) + np.finfo(float).eps))

#         return weight_tensor
import numpy as np

class Sgd(object):
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor 
        return weight_tensor

class SgdWithMomentum(object):
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0.
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.learning_rate * gradient_tensor + self.momentum_rate * self.v
        weight_tensor = weight_tensor - v
        self.v = v
        return weight_tensor

class Adam(object):
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0.
        self.r = 0.
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        self.k += 1
        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        return weight_tensor