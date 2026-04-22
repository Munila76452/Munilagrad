import numpy as np
from munilagrad.engine import value
from munilagrad.nn import conv2D ,Linear, flatten
import matplotlib.pyplot as plt

# print("1. Creating dummy image (Batch: 1, Channels: 3, Height: 5, Width: 5)...")
# dummy_image = value(np.random.randn(1, 3, 5, 5))

# print("2. Initializing conv2D layer...")
# # 3 input channels, 2 output filters, 3x3 kernel, padding=1
# layer = conv2D(inp_channel=3, out_channel=2, Kernel_size=3, padding=1)

# print("\n--- FORWARD PASS ---")
# output = layer(dummy_image)
# print(f"Input shape:  {dummy_image.data.shape}")
# print(f"Output shape: {output.data.shape}") 
# # Expected Output: (1, 2, 5, 5) because padding=1 keeps the spatial size the same!

# print("\n--- CALCULATING LOSS ---")
# # Sum all values to create a single scalar loss
# loss = output.sum()
# print(f"Loss value: {loss.data:.4f}")

# print("\n--- BACKWARD PASS ---")
# loss.backward()

# print("Checking gradients...")
# print(f"Gradient populated on Kernel? {'YES' if np.any(layer.w.grad) else 'NO'}")
# print(f"Kernel gradient shape: {layer.w.grad.shape}") # Should be (2, 3, 3, 3)

# print(f"Gradient populated on Bias? {'YES' if np.any(layer.b.grad) else 'NO'}")
# print(f"Bias gradient shape: {layer.b.grad.shape}") # Should be (1, 2, 1, 1)

# print("\nSUCCESS! munilagrad Conv2D is fully operational.")

# 1. CREATE THE TOY DATASET (4 Images, 5x5 pixels, 1 Channel)
# X_data = np.zeros((4, 1, 5, 5))
# X_data[0, 0, :, 2] = 1.0 # Image 0: Vertical Line (Target: 1)
# X_data[1, 0, 2, :] = 1.0 # Image 1: Horizontal Line (Target: -1)
# X_data[2, 0, :, 1] = 1.0 # Image 2: Vertical Line offset (Target: 1)
# X_data[3, 0, 1, :] = 1.0 # Image 3: Horizontal Line offset (Target: -1)

# Y_data = np.array([[1.0], [-1.0], [1.0], [-1.0]])

# X_train = value(X_data)
# Y_targets = value(Y_data)

# # 2. BUILD THE TINY CNN
# class TinyCNN:
#     def __init__(self):
#         self.conv = conv2D(inp_channel=1, out_channel=2, Kernel_size=3, padding=0)
#         self.flat = flatten()
#         self.fc = Linear(nin=18, nout=1) # 2 channels * 3 height * 3 width = 18

#     def __call__(self, x):
#         x = self.conv(x)
#         x = x.relu()
#         x = self.flat(x)
#         out = self.fc(x)
#         return out.tanh()

#     def parameters(self):
#         return self.conv.parameters() + self.fc.parameters()

# model = TinyCNN()

# # 3. THE TRAINING LOOP
# epochs = 50
# learning_rate = 0.05
# loss_history = []

# print("Starting Training...")
# for epoch in range(epochs):
#     # Forward Pass
#     predictions = model(X_train)
    
#     # Calculate Loss (Mean Squared Error)
#     diff = predictions - Y_targets
#     squared_error = diff * diff
#     total_loss = squared_error.sum() 
    
#     loss_val = total_loss.data.item() if np.isscalar(total_loss.data) else total_loss.data.flatten()[0]
#     loss_history.append(loss_val)
    
#     # Zero Gradients
#     for p in model.parameters():
#         p.grad = np.zeros_like(p.data)
        
#     # Backward Pass
#     total_loss.backward()
    
#     # Update Weights
#     for p in model.parameters():
#         p.data -= learning_rate * p.grad
        
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch:02d} | Loss: {loss_val:.4f}")

# print("\n--- FINAL PREDICTIONS ---")
# print("Target:   [ 1.0, -1.0,  1.0, -1.0]")
# print("Predicted:", predictions.data.flatten().round(2))

# # 4. PLOT THE LEARNING CURVE
# plt.figure(figsize=(8, 5))
# plt.plot(loss_history, marker='o', linestyle='-', linewidth=2)
# plt.title("munilagrad Learning Curve (Toy Dataset)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()
import time
import numpy as np
import matplotlib.pyplot as plt
from munilagrad.engine import value
from munilagrad.nn import conv2D, maxpool2D, flatten, Linear
from munilagrad.loss import MSELoss
from munilagrad.optim import SGD

# --- 1. Dataset Generation ---
# 4 samples of 6x6 images (1 channel)
X = np.zeros((4, 1, 6, 6))

# Vertical Lines
X[0, 0, :, 2:4] = 1.0
X[1, 0, :, 3:5] = 1.0
# Horizontal Lines
X[2, 0, 2:4, :] = 1.0
X[3, 0, 3:5, :] = 1.0

# Targets: 1.0 for Vertical, 0.0 for Horizontal
Y = np.array([
    [1.0], 
    [1.0], 
    [0.0], 
    [0.0]
])

# --- 2. Model Definition ---
class SimpleCNN:
    def __init__(self):
        self.conv1 = conv2D(inp_channel=1, out_channel=2, Kernel_size=3, stride=1, padding=0)
        self.pool = maxpool2D(Kernel_size=2, stride=2)
        # Note: Depending on your pooling implementation, nin might be 8. Adjust if necessary!
        self.fc = Linear(nin=8, nout=1) 

    def __call__(self, x):
        x = self.conv1(x).relu() 
        x = self.pool(x)      
        x = x.flatten()
        out = self.fc(x)
        return out

    def parameters(self):
        return self.conv1.parameters() + self.fc.parameters()

# --- 3. Training Setup ---
model = SimpleCNN()
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.05) 

epochs = 50
loss_history = []

print("Starting CNN Training Benchmarks")
print()

# ==========================================
# start time 
# ==========================================
start_time = time.time()

for epoch in range(epochs):
    # Forward Pass
    predictions = model(value(X))
    loss = criterion(predictions, Y)
    
    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update Weights
    optimizer.step()
    
    loss_history.append(loss.data.item())
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:02d} | Loss: {loss.data.item():.4f}")

# ==========================================
# end time
# ==========================================
end_time = time.time()
total_time = end_time - start_time

# --- 4. Benchmark Results ---
# print("\n" + "="*40)
print()
print(f"TRAINING COMPLETED IN: {total_time:.4f} seconds")
print()
# print("="*40 + "\n")
# --- 5. Final Output ---
print("Final Predictions")
final_preds = model(value(X))
for i in range(len(X)):
    label = "Vertical  " if Y[i][0] == 1.0 else "Horizontal"
    print(f"{label} Image -> Target: {Y[i][0]} | Prediction: {final_preds.data[i][0]:.4f}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, color='red', linewidth=2)
plt.title(f"Munilagrad CNN Training Loss\n(Execution Time: {total_time:.4f}s)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()