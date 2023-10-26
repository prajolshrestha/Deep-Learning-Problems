import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """ 
        Basically, Encoder is sequence of submodels that reduces dimension of data but,
        at the same time increases its number of features.

        latent space: Multivariate (Gaussian) distribution with mean and std.     

        Variational Auto-Encoder: learns latent space!
                                    (ie, learns parameter(mean and std) of multivariate distribution)

        Residual Block: combination of convolution and normalization  and it does not change the shape of image.
                        - used to increase number of features. 

        Attention Block: Think image as sequance of pixel, 
                         attention helps to relate pixel with each other.
    """
    def __init__(self, ):
        super().__init__(
            # (Batch_size, channel, height, width) --> (batch_size, 128, height, width)
            nn.Conv2d(3,128,kernel_size=3,padding=1), # Attention: No shape change! (because of padding and kernel size)

            #(batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128), 

            #(batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128), 
           
            # (batch_size, 128, height, width) ->(batch_size, 128, height/2, width/2)
            nn.Conv2d(128,128,kernel_size=3, stride=2, padding=0), # Attention: Shape changed!
        	
            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),  # Attention: feature size increased!

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256), 
            
            # (batch_size, 256, height/2, width/2) ->(batch_size, 256, height/4, width/4)
            nn.Conv2d(256,256,kernel_size=3, stride=2, padding=0), # Attention: Shape changed!

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),  # Attention: feature size increased!

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512), 
            
            # (batch_size, 512, height/4, width/4) ->(batch_size, 512, height/8, width/8)
            nn.Conv2d(512,512,kernel_size=3, stride=2, padding=0), # Attention: Shape changed!
           
            VAE_ResidualBlock(512, 512), 

            VAE_ResidualBlock(512, 512),  

            # (batch_size, 512, height/8, width/8) ->(batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),  

            # (batch_size, 512, height/8, width/8) ->(batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, height/8, width/8) ->(batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) ->(batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),

            # (batch_size, 512, height/8, width/8) ->(batch_size, 512, height/8, width/8)
            nn.SiLU(),

            # (batch_size, 512, height/8, width/8) ->(batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), # Attention:  shape changed!

            # (batch_size, 8, height/8, width/8) ->(batch_size, 8, height/8, width/8) 
            nn.Conv2d(8,8, kernel_size=1, padding=0)
        )


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, channel, height, weight)
        # noise = (batch_size, out_channel, height/8, weight/8)

        for module in self:
            # if a module has stride = (2,2), we pad in an asymetrical way.
            # pad = (pad_left, pad_right, pad_top, pad_bottom)
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            
            x = module(x)

        # (batch_size, 8, height/8, weight/8) -> two tensor of size (batch_size, 4, height/8, weight/8)
        mean, log_variance = nn.chunk(x, 2, dim=1)

        # (batch_size, 4, height/8, weight/8) -> (batch_size, 4, height/8, weight/8)
        log_variance = torch.clamp(log_variance, -30, 30)
        
        # (batch_size, 4, height/8, weight/8) -> (batch_size, 4, height/8, weight/8)
        #compute variance 
        variance = log_variance.exp()

        # (batch_size, 4, height/8, weight/8) -> (batch_size, 4, height/8, weight/8)
        # compute std
        stdev = variance.sqrt()

        # Z = N(0,1) ---> N(mean, std) = X
        x = mean + noise * stdev
        
        # scale x
        x *= 0.18215

        return x
