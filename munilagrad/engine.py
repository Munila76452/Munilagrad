import math
import numpy as np
from munilagrad.utils import im2col_loops,col2im_loops,img2col,col2img
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

  def sum(self):
    out = value(np.sum(self.data),(self,),'sum')
    
    def _backward():
      self.grad += value.unbroacasting(out.grad * np.ones_like(self.data),self.data.shape)
      
    out._backward = _backward
    return out
  
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
  
  def softmax(self, axis=-1):
    x = self.data
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted_x)
    probs = exps / np.sum(exps, axis=axis, keepdims=True)
    out = value(probs, (self,), 'softmax')
    
    def _backward():
        dout = out.grad
        sum_dout_probs = np.sum(dout * probs, axis=axis, keepdims=True)
        self.grad += probs * (dout - sum_dout_probs)
    
    out._backward = _backward
    return out
  
  def log(self):
    # adding  tiny epsilon to prevent log(0)
    out = value(np.log(self.data + 1e-15),(self,),'log')
    def _backward():
      self.grad += value.unbroacasting(out.grad * (1/(self.data + 1e-15)),self.data.shape)
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
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad
    
    out._backward = _backward
    return out
  
  def conv2D(self,weight,bias,stride=1,padding=0):
    x = self.data
    w = weight.data
    b = bias.data if bias else None
    
    if isinstance(stride,int):
      sh,sw = stride,stride
    else:
      sh,sw = stride
    
    if isinstance(padding,int):
      ph,pw = padding,padding
    else:
      ph,pw = padding
    
    N, Cin, Hin, Win = x.shape
    Cout, _, Kh, Kw = w.shape
    
    Hout = (Hin + 2 * ph - Kh) // sh + 1
    Wout = (Win + 2 * pw - Kw) // sw + 1
    
    # x_padded = np.pad(
    #   x,
    #   ((0,0),(0,0),(ph,ph),(pw,pw)),
    #   mode='constant'
    # )
    # out_data = np.zeros((N,Cout,Hout,Wout))
    
    # for n in range(N):
    #   for cout in range(Cout):
    #     for i in range(Hout):
    #       for j in range(Wout):
            
    #         h_start = i * sh
    #         w_start = j * sw
            
    #         patch = x_padded[
    #           n,
    #           :,
    #           h_start : h_start + Kh,
    #           w_start : w_start + Kw
    #         ]
    #         out_data[n,cout,i,j] = np.sum(
    #           patch * w[cout]
    #         )
    #         if b is not None:
    #           out_data[n,cout,i,j] += b[0,cout,0,0]
    X_col = img2col(x,(Kh,Kw),stride=(sh,sw),padding=(ph,pw))
    W_row = w.reshape(Cout,-1)
    out_col = W_row @ X_col
    
    if bias is not None:
      out_col += b.reshape(Cout,1)
    
    # out_data = out_col.reshape(Cout, N, Hout, Wout).transpose(1, 0, 2, 3)
    out_data = out_col.reshape(Cout, Hout, Wout, N).transpose(3, 0, 1, 2)
    
    children = (self,weight) if bias is None else (self,weight,bias)
    out = value(out_data,children,'conv2D')
  
    def _backward():
      # dout = out.grad
      # dx_padded = np.zeros_like(x_padded,dtype=np.float32)
      # dkernel = np.zeros_like(w,dtype=np.float32)
      
      # if bias is not None:
      #   dbias = np.sum(dout,axis=(0,2,3),keepdims=True).astype(np.float32)
      #   bias.grad += dbias
      
      # for n in range(N):
      #   for cout in range(Cout):
      #     for i in range(Hout):
      #       for j in range(Wout):
              
      #         h_start = i * sh
      #         w_start = j * sw
              
      #         x_slice = x_padded[
      #           n,
      #           :,
      #           h_start : h_start + Kh,
      #           w_start : w_start + Kw
      #         ]
      #         grad = dout[n,cout,i,j]
      #         dkernel[cout] += grad*x_slice
      #         dx_padded[n,:,h_start:h_start+Kh,w_start:w_start+Kw] += grad * w[cout]
              
      # if ph > 0 or pw > 0:
      #     dx = dx_padded[:, :, ph:-ph, pw:-pw]
      # else:
      #     dx = dx_padded
          
      # self.grad += dx
      # weight.grad += dkernel
      
      dout = out.grad
      dout_reshaped = dout.transpose(1,0,2,3).reshape(Cout,-1)
      if bias is not None:
        bias.grad += np.sum(dout,axis=(0,2,3),keepdims=True).astype(np.float32)
      
      dw_flattened = dout_reshaped @ X_col.T
      weight.grad += dw_flattened.reshape(w.shape)

      dX_col = W_row.T @ dout_reshaped

      self.grad += col2img(dX_col, x.shape, (Kh, Kw), stride=(sh, sw), padding=(ph, pw))
      
    out._backward = _backward
    return out
  
  def transposed_conv2D(self,weight,bias,stride=1,padding=0):
    x = self.data
    w = weight.data
    b = bias.data if bias else None
    
    if isinstance(stride,int):
      sh,sw = stride,stride
    else:
      sh,sw = stride
    if isinstance(padding,int):
      ph,pw = padding,padding
    else:
      ph,pw = padding
      
    N, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = w.shape
    
    H_out = (H_in - 1) * sh - 2 * ph + K_h
    W_out = (W_in - 1) * sw - 2 * pw + K_w
    
    out_data = np.zeros((N,C_out,H_out,W_out))
    
    for n in range(N):
      for c_in in range(C_in):
        for c_out in range(C_out):
          for i in range(H_in):
            for j in range(W_in):
              # calculating therotical bound
              h_start = i * sh - ph
              w_start = j * sw - pw
              h_end = h_start + K_h
              w_end = w_start + K_w
              
              # handling padding - constraint to valid bound
              out_h_start = max(0,h_start)
              out_w_start = max(0,w_start)
              out_h_end = min(H_out,h_end)
              out_w_end = min(W_out,w_end)
              
              # handling padding - crop the kernal stamp
              k_h_start = out_h_start - h_start
              k_w_start = out_w_start - w_start
              k_h_end = K_h - (h_end - out_h_end)
              k_w_end = K_w - (w_end - out_w_end)
              
              out_data[n,c_out,out_h_start:out_h_end,out_w_start:out_w_end] += x[n,c_in,i,j] * w[c_in,c_out,k_h_start:k_h_end,k_w_start:k_w_end]
    if b is not None:
      out_data += b.reshape(1,C_out,1,1)
    children = (self,weight) if bias is None else (self,weight,bias)
    out = value(out_data,children,'transposed_conv2D')
    
    def _backward():
      dout = out.grad
      dx = np.zeros_like(x,dtype=float)
      dw = np.zeros_like(w,dtype=float)
      
      if bias is not None:
        bias.grad += np.sum(dout,axis=(0,2,3),keepdims=True).astype(float)
        
      for n in range(N):
        for c_in in range(C_in):
          for c_out in range(C_out):
            for i in range(H_in):
              for j in range(W_in):
                
                h_start = i * sh - ph
                w_start = j * sw - pw
                h_end = h_start + K_h
                w_end = w_start + K_w
                
                out_h_start = max(0, h_start)
                out_w_start = max(0, w_start)
                out_h_end = min(H_out, h_end)
                out_w_end = min(W_out, w_end)
                
                k_h_start = out_h_start - h_start
                k_w_start = out_w_start - w_start
                k_h_end = K_h - (h_end - out_h_end)
                k_w_end = K_w - (w_end - out_w_end)
                
                dout_slice = dout[n,c_out,out_h_start:out_h_end,out_w_start:out_w_end]
                k_slice = w[c_in, c_out, k_h_start:k_h_end, k_w_start:k_w_end]
                
                dx[n,c_in,i,j] += np.sum(dout_slice * k_slice)
                dw[c_in,c_out,k_h_start:k_h_end,k_w_start:k_w_end] += x[n, c_in, i, j] * dout_slice
                
      self.grad += dx
      weight.grad += dw
      
    out._backward = _backward
    return out
  
  def flatten(self):
    out_data =  self.data.reshape(self.data.shape[0],-1)
    out = value(out_data,(self,),'flatten')
    
    def _backward():
      self.grad += out.grad.reshape(self.data.shape)
      
    out._backward = _backward
    return out
  
  def maxPool(self,Kernel_size,stride=1,padding=0):
    x = self.data
    
    if isinstance(Kernel_size,int):
      Kh , Kw = Kernel_size,Kernel_size
    else:
      Kh,Kw = Kernel_size
      
    if isinstance(stride,int):
      sh , sw = stride,stride
    else:
      sh,sw = stride
    
    if isinstance(padding,int):
      ph,pw = padding,padding
    else:
      ph,pw = padding
      
    N,C,H_in,W_in = x.shape
    
    H_out = ((H_in + 2 * ph - Kh) // sh) + 1
    W_out = ((W_in + 2 * pw - Kw) // sw) + 1
    
    # x_padded = np.pad(
    #   x,
    #   ((0,0),(0,0),(ph,ph),(pw,pw)),
    #   mode='constant',
    #   constant_values = -np.inf
    # )
    
    # out_data = np.zeros((N,C,H_out,W_out))
    
    # for n in range(N):
    #   for c in range(C):
    #     for i in range(H_out):
    #       for j in range(W_out):
            
    #         h_start = i * sh
    #         w_start = j * sw
            
    #         patch = x_padded[
    #           n,
    #           c,
    #           h_start : h_start + Kh,
    #           w_start : w_start + Kw
    #         ]
            
    #         out_data[n,c,i,j] = np.max(patch)
    X_col = img2col(x, (Kh, Kw), stride=(sh, sw), padding=(ph, pw))
    
    X_col_reshaped = X_col.reshape(C, Kh * Kw, -1)

    out_col = np.max(X_col_reshaped, axis=1) 
    max_indices = np.argmax(X_col_reshaped, axis=1) 
    
    # out_data = out_col.reshape(C, N, H_out, W_out).transpose(1, 0, 2, 3)
    out_data = out_col.reshape(C, H_out, W_out, N).transpose(3, 0, 1, 2)
    
    out = value(out_data,(self,),'maxpool')
    
    def _backward():
      dout = out.grad
      
      # dx_padded = np.zeros_like(x_padded,dtype=float)
      
      # for n in range(N):
      #   for c in range(C):
      #     for i in range(H_out):
      #       for j in range(W_out):
      #         h_start = i * sh
      #         w_start = j * sw
              
      #         patch = x_padded[
      #           n,
      #           c,
      #           h_start : h_start + Kh,
      #           w_start : w_start + Kw
      #         ]
      #         idx = np.argmax(patch)
      #         max_idx_h , max_idx_w = np.unravel_index(idx,patch.shape)
      #         dx_padded[n,c,h_start+max_idx_h,w_start+max_idx_w] += dout[n,c,i,j]
              
      # if ph > 0 or pw > 0:
      #   dx = dx_padded[:,:,ph:-ph,pw:-pw]
      # else:
      #   dx = dx_padded
        
      # self.grad += dx
      dout_flat = dout.transpose(1, 0, 2, 3).reshape(C, -1)

      dX_col_reshaped = np.zeros_like(X_col_reshaped)

      c_idx = np.arange(C).reshape(-1, 1) 
      col_idx = np.arange(dX_col_reshaped.shape[2]) 
      
      dX_col_reshaped[c_idx, max_indices, col_idx] = dout_flat
      
      dX_col = dX_col_reshaped.reshape(C * Kh * Kw, -1)
      
      self.grad += col2img(dX_col, x.shape, (Kh, Kw), stride=(sh, sw), padding=(ph, pw))
    out._backward = _backward
    return out
  
  def global_avg_pooling(self):
    B,C,H,W = self.data.shape
    out_data = np.mean(self.data,axis=(2,3))
    out = value(out_data,(self,),'global_avg_pooling')
    
    def _backward():
      num_pixel = H * W
      grad_distribution = out.grad / num_pixel
      grad_reshaped = grad_distribution.reshape(B,C,1,1)
      self.grad += grad_reshaped * np.ones_like(self.data)
    
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
    