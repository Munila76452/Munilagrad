from munilagrad.engine import value 
from munilagrad.nn import MLP
from munilagrad.viz import draw_dot
import numpy as np

# dataset
xs = np.array([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
])

ys = np.array([[1.0], [-1.0], [-1.0], [1.0]])

# model
model = MLP(3, [4, 4, 1])

lr = 0.01   # small learning rate (important)

for k in range(50):

    # wrap inputs
    x = value(xs)
    y_true = value(ys)

    # forward
    y_pred = model(x)

    # loss (MSE)
    diff = y_pred - y_true
    loss = diff * diff   # (batch,1)

    # mean loss
    loss_mean = value(np.mean(loss.data), (loss,), 'mean')

    # manual backward for mean
    def _backward():
        loss.grad += np.ones_like(loss.data) / loss.data.size

    loss_mean._backward = _backward

    # zero gradients
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)

    # backward
    loss_mean.backward()

    # gradient clipping (optional but safe)
    for p in model.parameters():
        p.grad = np.clip(p.grad, -1, 1)

    # update
    for p in model.parameters():
        p.data += -lr * p.grad

    print(k, loss_mean.data)
# DRAW THE GRAPH ONCE AFTER TRAINING
print("Rendering computational graph...")
dot = draw_dot(loss)
dot.render('mlp_graph', format='svg', view=True)