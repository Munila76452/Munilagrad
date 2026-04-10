from munilagrad.engine import value 
from munilagrad.nn import MLP
from munilagrad.viz import draw_dot
import numpy as np
# a = value([[1],[2],[3]])   # (3,1)
# b = value(np.ones((3,4)))  # (3,4)

# c = a / b
# c.grad = np.ones((3,4))

# c._backward()
# print(a.grad)
# print(a.grad.shape)
import numpy as np

a = value(np.random.randn(3,4))
b = value(np.random.randn(4,2))

c = a.matmul(b)
c.grad = np.ones((3,2))

c._backward()

print("a.grad shape:", a.grad.shape)  # (3,4)
print("b.grad shape:", b.grad.shape)  # (4,2)
