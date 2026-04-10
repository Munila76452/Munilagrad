import math
import numpy as np
class value:
  def __init__(self,data,children=(),_op='', label=''):    
    self.data = np.array(data,dtype=float)
    self.grad = np.zeros_like(self.data)
    
    self._prev = set(children)
    self._backward = lambda:None
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"value(data={self.data})"
  
  @staticmethod
  def unbroacasting(grad,shape):
    '''
    when the forward pass happens then the numpy will internally handles broadcasting
    for ex - (3,1) becomes (3,4) 
    then on backward pass  we dont want (3,4) we want (3,1) to do the steady cal
    so we uses unbroadcasting
    '''
    while (len(grad.shape) > len(shape)):
      grad = grad.sum(axis=0)
    for i , dim in enumerate(shape):
      if dim == 1:
        grad = grad.sum(axis=i,keepdims=True)
        
    return grad
  
  def __add__(self,other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data + other.data,(self,other),'+')
    def _backward():
      self.grad += value.unbroacasting(out.grad,self.data.shape)
      other.grad += value.unbroacasting(out.grad,other.data.shape)
    out._backward = _backward
    return out

  def __mul__(self,other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data * other.data,(self,other),'*')
    def _backward():
      self.grad += value.unbroacasting(other.data * out.grad,self.data.shape)
      other.grad += value.unbroacasting(self.data * out.grad,other.data.shape)
    out._backward = _backward
    return out

  def __pow__(self,other):
    assert isinstance(other,(int,float))
    out = value(self.data**other,(self,),f'**{other}')
    def _backward():
      self.grad += value.unbroacasting(other * (self.data ** (other-1)) * out.grad,self.data.shape)
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other
  
  def __rmul__(self,other):
    return self * other

  def __truediv__(self,other):
    return self * other ** -1

  def __neg__(self):
    return self * -1

  def __sub__(self,other):
    return self + (-other)

  def __rsub__(self, other):
    other = other if isinstance(other, value) else value(other)
    return other - self

  def tanh(self):
    x = self.data
    t = (np.exp(2*x)-1) / (np.exp(2*x)+1)
    out = value(t,(self,),'tanh')
    def _backward():
      self.grad += value.unbroacasting((1-t**2) * out.grad,self.data.shape)
    out._backward = _backward
    return out

  def relu(self):
    out = value(np.maximum(0,self.data),(self,),'RELU')
    def _backward():
      self.grad += value.unbroacasting((out.data > 0) * out.grad,self.data.shape)
    out._backward = _backward
    return out
  
  def exp(self):
    x = self.data
    out = value(np.exp(x),(self,),'exp')
    def _backward():
      self.grad += value.unbroacasting(out.data * out.grad,self.data.shape)
    out._backward = _backward
    return out
  
  def matmul(self,other):
    other = other if isinstance(other,value) else value(other)
    
    out = value(self.data @ other.data , (self,other),'matmul')
    def _backward():
      self.grad = out.grad @ other.data.T
      other.grad = self.data.T @ out.grad
    
    out._backward = _backward
    return out
  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = np.ones_like(self.data)

    for node in reversed(topo):
        node._backward()
    