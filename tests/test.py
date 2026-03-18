from munilagrad.engine import value
from munilagrad.nn import MLP
from munilagrad.viz import draw_dot
xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

ys = [1.0,-1.0,-1.0,1.0]

n = MLP(3,[4,4,1])
for k in range(20):
    ypred = [n([value(xi) for xi in x]) for x in xs]
    loss = sum(((yout - value(ygt))**2 for ygt,yout in zip(ys,ypred)), value(0))
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
# DRAW THE GRAPH ONCE AFTER TRAINING
print("Rendering computational graph...")
dot = draw_dot(loss)
dot.render('mlp_graph', format='svg', view=True)