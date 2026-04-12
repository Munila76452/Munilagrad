import numpy as np
import matplotlib.pyplot as plt  # 1. Add this import
from munilagrad.engine import value
from munilagrad.nn import MLP
from munilagrad.loss import MSELoss
from munilagrad.optim import SGD

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

model = MLP(nin=2, nout=[4, 1])
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

epochs = 500
loss_history = [] 

for epoch in range(epochs):
    predictions = model(value(X))
    loss = criterion(predictions, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.data.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data.item():.4f}")

print("\n--- Final Predictions ---")
final_preds = model(value(X))
for i in range(len(X)):
    print(f"Input: {X[i]} | Target: {Y[i]} | Pred: {final_preds.data[i]}")

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, color='blue', linewidth=2)
plt.title("Munilagrad XOR Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()