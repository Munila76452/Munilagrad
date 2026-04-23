from munilagrad.engine import value 
from munilagrad.nn import MLP
from munilagrad.viz import draw_dot
import numpy as np

# 1. Create a tiny 1x1x2x2 input (Batch, Channels, Height, Width)
x_data = np.array([[[[1.0, 2.0], 
                     [3.0, 4.0]]]])
x = value(x_data, label='x')

# 2. Create a 1x1x3x3 kernel (C_in, C_out, K_h, K_w)
w_data = np.ones((1, 1, 3, 3))
w = value(w_data, label='w')

# 3. Forward Pass: Stride 2, Padding 0
out = x.transposed_conv2D(w, bias=None, stride=2, padding=0)

# 4. Create a dummy loss (sum of all elements)
loss = out.sum()

# 5. Backward Pass
loss.backward()

# 6. Check the results
print("Output Shape:", out.data.shape) # Should be (1, 1, 5, 5)
print("\nInput Gradients (dx):\n", x.grad)
