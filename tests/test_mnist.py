import numpy as np
from munilagrad.engine import value
from munilagrad.loss import CrossEntropyLoss
from munilagrad.optim import SGD
from munilagrad.mnist_data import fetch_mnist, DataLoader

class MunilaNet:
    def __init__(self):
        self.conv1_w = value(np.random.randn(4, 1, 3, 3) * np.sqrt(2.0 / (1 * 3 * 3)))
        self.conv1_b = value(np.zeros((1, 4, 1, 1)))
        
        self.fc1_w = value(np.random.randn(784, 10) * np.sqrt(2.0 / 784))
        self.fc1_b = value(np.zeros((1, 10)))

    def parameters(self):
        return [self.conv1_w, self.conv1_b, self.fc1_w, self.fc1_b]

    def forward(self, x):
        x = value(x)
        x = x.conv2D(self.conv1_w, self.conv1_b, stride=1, padding=1)
        x = x.relu()
        x = x.maxPool(Kernel_size=2, stride=2, padding=0)
        x = x.flatten()
        out = x.matmul(self.fc1_w) + self.fc1_b
        return out

    def save_weights(self, filename="munilanet.npz"):
        """Saves the trained NumPy arrays to the hard drive."""
        weights = {
            "conv1_w": self.conv1_w.data,
            "conv1_b": self.conv1_b.data,
            "fc1_w": self.fc1_w.data,
            "fc1_b": self.fc1_b.data
        }
        np.savez(filename, **weights)
        print(f"\n Model weights successfully saved to {filename}")

    def load_weights(self, filename="munilanet.npz"):
        """Loads trained weights back into the model."""
        with np.load(filename) as data:
            self.conv1_w.data = data["conv1_w"]
            self.conv1_b.data = data["conv1_b"]
            self.fc1_w.data = data["fc1_w"]
            self.fc1_b.data = data["fc1_b"]
        print(f" Model weights loaded from {filename}")
        
if __name__ == "__main__":
    print("Downloading and Parsing MNIST Dataset...")
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    
    train_loader = DataLoader(X_train, Y_train, batch_size=32, shuffle=True)
    
    model = MunilaNet()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    total_params = sum(p.data.size for p in model.parameters())
    print("-" * 30)
    print(f"MODEL ARCHITECTURE:")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 30)

    epochs = 3
    print("\nStarting Training Loop...")
    print("-" * 30)
    
    for epoch in range(epochs):
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            logits = model.forward(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx:04d} | Loss: {loss.data:.4f}")
                
    print("\nTraining Completed")
    
    model.save_weights()

    # infernce
    print("\n" + "="*30)
    print("RUNNING PREDICTIONS ON TEST DATA")
    print("="*30)

    for i in range(5):
        # Slice to keep shape at (1, 1, 28, 28)
        test_image = X_test[i:i+1] 
        true_label = Y_test[i]
        
        # Forward pass (no backward or optimizer step!)
        logits = model.forward(test_image)
        probs = logits.softmax()
        
        # Get the prediction and confidence
        predicted_digit = np.argmax(probs.data, axis=1)[0]
        confidence = np.max(probs.data, axis=1)[0] * 100
        
        if predicted_digit == true_label:
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
            
        print(f"Image {i} | True: {true_label} | Predicted: {predicted_digit} | Confidence: {confidence:.1f}% | {status}")

    # Visualizer for the final image
    print("\nHere is what the network just looked at (Image 4):")
    pixels = X_test[4].reshape(28, 28)
    for row in pixels:
        line = "".join(["██" if p > 0.5 else "░░" if p > 0.1 else "  " for p in row])
        print(line)