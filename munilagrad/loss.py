from munilagrad.engine import value
import numpy as np
class MSELoss:
    
    def __call__(self,prediction,target):
        diff = prediction - target
        squared_diff = diff ** 2
        
        n_element = np.prod(prediction.data.shape)
        
        loss = squared_diff.sum() * (1.0 / n_element)
        
        return loss
        