from munilagrad.engine import value
import numpy as np
class MSELoss:
    
    def __call__(self,prediction,target):
        diff = prediction - target
        squared_diff = diff ** 2
        
        n_element = np.prod(prediction.data.shape)
        
        loss = squared_diff.sum() * (1.0 / n_element)
        
        return loss

class CrossEntropyLoss:
    
    def __call__(self, logits,targets):
        
        '''
        logits -> a raw prediction having size(batch_size,num_class)
        target -> A standard NumPy array of the true class indices. Shape: (Batch_Size,)
        Example: [5, 0, 4] means image 1 is a 5, image 2 is a 0, image 3 is a 4.
        '''
        x = logits.data
        N = x.shape[0]
        
        shifted_logits = x - np.max(x,axis=1,keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits,axis=1,keepdims=True)
        
        correct_prob = probs[np.arange(N),targets]
        
        loss_data = - np.sum(np.log(correct_prob +  1e-7)) / N
        
        out = value(loss_data,(logits,),'CrossEntropy')
        
        def _backward():
            dx = probs.copy()
            dx[np.arange(N),targets] -= 1
            dx = dx / N
            logits.grad += dx * out.grad
        
        out._backward = _backward
        return out