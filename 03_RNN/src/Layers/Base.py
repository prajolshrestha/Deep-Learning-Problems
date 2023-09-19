from typing import Any
import numpy as np

class BaseLayer(object):

    def __init__(self) -> None:
        
        self.trainable = False
        self.weights = []
        self.testing_phase = False

    def  forward(self):
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:

        return self.forward(*args)