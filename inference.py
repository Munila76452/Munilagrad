import numpy as np
from munilagrad.engine import value
from munilagrad.mnist_data import fetch_mnist

class MunilaNet:
    def __init__(self):
        self.conv1_w = value(np.random.randn(4, 1, 3, 3) * np.sqrt(2.0 / (1 * 3 * 3)))
        self.conv1_b = value(np.zeros((1, 4, 1, 1)))
        self.fc1_w = value(np.random.randn(784, 10) * np.sqrt(2.0 / 784))
        self.fc1_b = value(np.zeros((1, 10)))

    def forward(self, x):
        x = value(x)
        x = x.conv2D(self.conv1_w, self.conv1_b, stride=1, padding=1)
        x = x.relu()
        x = x.maxPool(Kernel_size=2, stride=2, padding=0)
        x = x.flatten()
        out = x.matmul(self.fc1_w) + self.fc1_b
        return out

    def load_weights(self, filename="munilanet.npz"):
        """Loads trained weights back into the model."""
        with np.load(filename) as data:
            self.conv1_w.data = data["conv1_w"]
            self.conv1_b.data = data["conv1_b"]
            self.fc1_w.data = data["fc1_w"]
            self.fc1_b.data = data["fc1_b"]
        print(f" Model weights successfully loaded from {filename}")

if __name__ == "__main__":
    print("Loading MNIST Test Dataset...")
    _, _, X_test, Y_test = fetch_mnist()
    
    model = MunilaNet()
    
    try:
        model.load_weights("munilanet.npz")
    except FileNotFoundError:
        print(" Error: Could not find 'munilanet.npz'. Run train.py first to generate the weights!")
        exit()

    print("\n" + "="*30)
    print("RUNNING PREDICTIONS ON TEST DATA")
    print("="*30)

    # Let's look at the first 5 images in the unseen Test set
    for i in range(5):
        # Slice to keep shape at (1, 1, 28, 28)
        test_image = X_test[i:i+1] 
        true_label = Y_test[i]
        
        # Forward pass (no backward or optimizer step needed for inference!)
        logits = model.forward(test_image)
        probs = logits.softmax()
        
        # Get the prediction and confidence
        predicted_digit = np.argmax(probs.data, axis=1)[0]
        confidence = np.max(probs.data, axis=1)[0] * 100
        
        if predicted_digit == true_label:
            status = " CORRECT"
        else:
            status = " WRONG"
            
        print(f"Image {i} | True: {true_label} | Predicted: {predicted_digit} | Confidence: {confidence:.1f}% | {status}")

    # Visualizer for the final image
    print("\nHere is what the network just looked at (Image 4):")
    pixels = X_test[4].reshape(28, 28)
    for row in pixels:
        line = "".join(["██" if p > 0.5 else "░░" if p > 0.1 else "  " for p in row])
        print(line)