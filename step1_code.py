import numpy as np
import matplotlib.pyplot as plt

def tent_map(x, r=1.5):
    """Tent Map Function"""
    return r * x if x < 0.5 else r * (1 - x)

def logistic_map(x, r=3.7):
    """Logistic Map Function"""
    return r * x * (1 - x)

def generate_chaotic_sequence(length, seed=0.45):
    """Generate Chaotic Key Sequence using Tent Map + Logistic Map"""
    x = seed
    chaotic_seq = []
    for _ in range(length):
        x = tent_map(logistic_map(x))  # Combining Tent & Logistic Maps
        chaotic_seq.append(x)
    return np.array(chaotic_seq)

# Generate a 256-bit chaotic key sequence
chaotic_key = generate_chaotic_sequence(256)

# Plot the chaotic sequence
plt.plot(chaotic_key, marker='o', linestyle='dashed', color='blue')
plt.title("Chaotic Key Sequence")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# Print first 10 values for reference
print("First 10 Chaotic Key Values:", chaotic_key[:10])
def chaotic_to_binary(chaotic_seq):
    """Convert chaotic sequence to a binary key"""
    binary_key = ""
    for value in chaotic_seq:
        # Convert to 8-bit binary (scaled between 0-255)
        binary_val = format(int(value * 255), '08b')
        binary_key += binary_val
    return binary_key

# Convert the chaotic sequence into a 256-bit binary key
binary_key = chaotic_to_binary(chaotic_key)

# Print first 32 bits (for reference)
print("Binary Key (First 32 bits):", binary_key[:32])
print("Total Key Length:", len(binary_key), "bits")
from scipy.stats import entropy
import collections

def calculate_entropy(binary_key):
    """Calculate Shannon Entropy of the binary key"""
    counts = collections.Counter(binary_key)  # Count 0s and 1s
    probs = [count / len(binary_key) for count in counts.values()]
    return entropy(probs, base=2)

def bit_distribution(binary_key):
    """Check distribution of 0s and 1s"""
    zeros = binary_key.count('0')
    ones = binary_key.count('1')
    return zeros, ones

# Calculate entropy
entropy_value = calculate_entropy(binary_key)

# Check bit distribution
zeros, ones = bit_distribution(binary_key)

print(f"Shannon Entropy: {entropy_value:.4f} (Max = 1.0 for perfect randomness)")
print(f"Bit Distribution: 0s = {zeros}, 1s = {ones}")
