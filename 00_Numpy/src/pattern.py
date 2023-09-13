"""
Created on WED September 13 13:40:15 2023

@author: Prajol Shrestha
"""

import numpy as np
import matplotlib.pyplot as plt

"""
    Goal: Create different types of patterns using Numpy only (No loops allowed).
            a. Checkerboard
            b. Circle
            c. Spectrum

    [Hint: np.tile(), np.arange(), np.zeros(), np.ones(), np.concatenate() and np.expand dims()]  
"""

class Checker:
    """
        Create a checkerboard pattern with adaptable tile and size resolution. 

        Hint: You might want to start with a fixed tile size and adapt later on. 
              For simplicity we assume that the resolution is divisible by the tile size without remainder.
    """

    def __init__(self,resolution, tile_size):
        '''
            Constructor
            Input: resolution:- number of pixels in each dimension
                   tile-size :- number of pixel an individual tile has in each dimension
        '''
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        '''
            Create pattern using numpy functions
        '''

        # First make a tile
        size = int(self.tile_size * 2)
        tile = np.zeros((size,size), dtype = int)
        tile[self.tile_size:, :self.tile_size] = 1
        tile[:self.tile_size, self.tile_size:] = 1
        #print(tile)

        # Repeat tile 
        num = int(self.resolution / (self.tile_size*2))
        self.output = np.tile(tile, (num,num))
        b = self.output.copy() 

        return b

    def show(self):
        '''
            Visualization function
        '''
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position):
        '''
            Constructor
        '''
        self.resolution = resolution
        self.radius = radius
        self.position = position # (x,y) coordinates of circle center
        

    def draw(self):
        '''
            Create pattern using numpy functions
        '''
        res = self.resolution
        x,y = np.meshgrid(np.arange(res), np.arange(res))# Create grid in 2D space (represents pixel position)
        distance = np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)# calculate euclidean distance from center position

        self.output= distance <= self.radius # compare distance with radius
        a = self.output.copy()        

        return a

    def show(self):
        '''
            Visualization function
        '''

        plt.imshow(self.draw(), cmap='gray')
        plt.show()

    
class Spectrum:

    def __init__(self, resolution):
        '''
            Constructor
        '''
        self.resolution = resolution
        

    def draw(self):
        '''
            Create pattern using numpy functions
        '''
        res = self.resolution
        
        spectrum = np.zeros([res,res,3])
        spectrum[:,:,0] = np.linspace(0,1,res) # each row ma 0 to 1 value halne
        spectrum[:,:,1] = np.linspace(0,1,res).reshape(res,1)
        spectrum[:,:,2] = np.linspace(1,0,res)
        self.output = spectrum 
        #print(spectrum[:,:,1])
        #print(spectrum[:,:,2])

        a = self.output.copy()        
        return a

        

    def show(self):
        '''
            Visualization function
        '''
        plt.imshow(self.draw())
        plt.show()

    