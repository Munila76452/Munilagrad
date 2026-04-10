from .engine import value
import random
import numpy as np
# class Neuron:

#   def __init__(self,nin):
#     self.w = [value(random.uniform(-1,1)) for _ in range(nin)]
#     self.b = value(random.uniform(-1,1))

#   def __call__(self,x):
#     act = sum((wi*xi for wi,xi in zip(self.w,x)) , self.b)
#     out = act.tanh()
#     return out

#   def parameters(self):
#     return self.w + [self.b]

# class layer:
#   def __init__(self,nin,nout):
#     self.neurons = [Neuron(nin) for _ in range(nout)]

#   def __call__(self,x):
#     outs = [n(x) for n in self.neurons]
#     return outs[0] if len(outs) == 1 else outs

#   def parameters(self):
#     return [p for neuron in self.neurons for p in neuron.parameters()]
    
# class MLP:
#   def __init__(self,nin,nouts):
#     sz = [nin] + nouts
#     self.layers = [layer(sz[i], sz[i+1]) for i in range(len(nouts))]

#   def __call__(self,x):
#     for layer in self.layers:
#       x = layer(x)
#     return x

#   def parameters(self):
#     return [p for layer in self.layers for p in layer.parameters()]

class Linear:
  def __init__(self,nin,nout):
    self.W = value(np.random.randn(nin,nout)*0.1)
    self.b = value(np.zeros((1,nout)))
    
  def __call__(self,x):
    return x.matmul(self.W) + self.b
  
  def parameters(self):
    return [self.W,self.b]
  
class MLP:
  def __init__(self,nin,nout):
    sz = [nin] + nout
    self.layers = [Linear(sz[i],sz[i+1]) for i in range(len(nout))]
    
  def __call__(self,x):
    for i , layer in enumerate(self.layers):
      x = layer(x)
      if i != len(self.layers)-1:
        x = x.relu()
        
    return x

  def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
      