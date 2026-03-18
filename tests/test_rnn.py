import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from munilagrad.engine import value # Assuming 'value' is in engine.py
from munilagrad.rnn import RNNCell
from munilagrad.viz import draw_dot
random.seed(42) 

rnn = RNNCell(input_size=2, hidden_size=3, output_size=1)

sequence = [
    [value(1.0), value(2.0)], # t=1
    [value(0.5), value(-1.0)],# t=2
    [value(-2.0), value(1.0)] # t=3
]

h_current = [value(0.0) for _ in range(3)]

print("Running Forward Pass--")
for t, x_t in enumerate(sequence):
    y_t, h_current = rnn.forward(x_t, h_current)
    print(f"Time Step {t+1} output: {y_t[0].data:.4f}")

# Target for the final prediction (at t=3) is 1.0
target = value(1.0)

# Calculate Mean Squared Error (MSE) loss on the final output only
final_prediction = y_t[0]
loss = (final_prediction - target)**2

print(f"\nFinal Loss: {loss.data:.4f}")

# backaward pass bptt
print("\nRunning BPTT---")
loss.backward()

# Let's look at the gradient for a specific weight in the hidden-to-hidden matrix
sample_weight = rnn.W_hh[0][0]
print(f"Gradient for W_hh[0][0]: {sample_weight.grad:.4f}")

# gradient descent update
learning_rate = 0.01
for p in rnn.parameters():
    p.data -= learning_rate * p.grad
    
print("Weights updated successfully!")
# to vizualize the graph
print("Rendering computational graph...")
dot = draw_dot(loss)
dot.render('rnn_graph', format='svg', view=True)
