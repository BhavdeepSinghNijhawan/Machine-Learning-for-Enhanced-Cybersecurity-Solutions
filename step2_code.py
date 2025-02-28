import numpy as np
from scipy.stats import entropy

def tent_map(x, r=1.99):  # Increased chaos factor
    return r * x if x < 0.5 else r * (1 - x)

def logistic_map(x, r=3.99):  # Maximum chaos near edge
    return r * x * (1 - x)

def generate_chaotic_sequence(length, seed=0.7345823):
    x = seed
    chaotic_seq = []
    for _ in range(length):
        for _ in range(5):  # More iterations for better mixing
            x = logistic_map(x)
            x = tent_map(x)
        chaotic_seq.append(x)
    return np.array(chaotic_seq)

# Generate the chaotic key
chaotic_key = generate_chaotic_sequence(256)

# Convert chaotic key to binary
binary_key = ''.join(['1' if x > np.mean(chaotic_key) else '0' for x in chaotic_key])

# Compute Shannon entropy
shannon_entropy = entropy([binary_key.count('0') / len(binary_key),
                           binary_key.count('1') / len(binary_key)], base=2)

# Count number of 0s and 1s
zeros = binary_key.count('0')
ones = binary_key.count('1')

# Print Analysis
print(f"First 10 Chaotic Key Values: {chaotic_key[:10]}")
print(f"Binary Key (First 32 bits): {binary_key[:32]}")
print(f"Total Key Length: {len(binary_key)} bits")
print(f"Shannon Entropy: {shannon_entropy:.4f} (Max = 1.0 for perfect randomness)")
print(f"Bit Distribution: 0s = {zeros}, 1s = {ones}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Generate a dataset of random keys (Weak = 0, Strong = 1)
def generate_dataset(samples=5000, length=256):
    X = np.random.rand(samples, length)  # Random keys
    y = np.random.randint(0, 2, samples)  # Labels (0 = Weak, 1 = Strong)
    return X, y

# Create dataset
X, y = generate_dataset()

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(256,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output (Weak = 0, Strong = 1)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Test the model on YOUR chaotic key
is_secure = model.predict(np.array([chaotic_key]))[0][0]
print(f"🔐 Key Security: {'Strong ✅' if is_secure > 0.5 else 'Weak ❌'}")
