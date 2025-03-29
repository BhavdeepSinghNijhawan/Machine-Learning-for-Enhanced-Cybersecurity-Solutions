# Machine Learning for Enhanced Cybersecurity Solutions

<h1 align="center">
  <img src="https://github.com/user-attachments/assets/e499d2f2-317a-4be3-ab76-6f752dcae069" alt="Project Workflow" width="500" />
</h1>

## Project Overview

This project implements a robust image encryption system combining cutting-edge cryptographic algorithms with machine learning techniques. Our solution integrates GANs (Generative Adversarial Networks), Kyber, NTRU, SHA, and AES to create a multi-layered security framework, enhanced with Shor's key and epoch-based security parameters.

## Key Features

- **Hybrid Encryption**: Combines post-quantum cryptography (Kyber, NTRU) with traditional encryption (AES)
- **GAN Integration**: Uses Generative Adversarial Networks for secure key generation and distribution
- **Quantum-Resistant**: Incorporates Shor's algorithm principles for future-proof security
- **Image Protection**: Specialized encryption for visual data with integrity checks via SHA
- **Epoch-Based Security**: Dynamic key rotation system for enhanced protection

## Visual Demonstration

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/d5e1f8fb-6452-40b6-ab3e-f91d30be2960" alt="Original Image" width="200"/>
        <br><em>Original Image (CIFAR-10 Sample)</em>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/5802457e-0580-47ec-afc6-bd6da04ae5f2" alt="Encrypted Image" width="200"/>
        <br><em>Encrypted Output</em>
      </td>
    </tr>
  </table>
</div>

## Technical Architecture

1. **Key Generation**:
   - Kyber/NTRU for post-quantum key pairs
   - GAN-assisted key strengthening
   - Epoch-based key rotation

2. **Encryption Process**:
   - AES-256 for bulk encryption
   - SHA-3 for integrity checks
   - Custom image transformation layers

3. **Security Enhancements**:
   - Shor's algorithm principles
   - Neural network-based anomaly detection
   - Adversarial training for resistance against ML attacks

## Implementation Details

```python
# Sample code structure (conceptual)
def hybrid_encrypt(image, public_key):
    # Generate session key using GAN
    session_key = gan_key_generator()
    
    # Post-quantum encapsulation
    ciphertext, shared_secret = kyber.encaps(public_key)
    
    # AES encryption with derived key
    encrypted_image = aes_encrypt(image, session_key)
    
    # Apply additional transformations
    final_output = apply_epoch_transforms(encrypted_image)
    
    return final_output, ciphertext
```

## REFERENCES

1. GitHub: https://gist.github.com/twheys/4e83567942172f8ba85058fae6bfeef5
2. WIKIPEDIA: https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
___
## CONTRIBUTORS

- Assistant Professor Dr. M. Jhamb (mansi.jhamb@ipu.ac.in), University School of Information, Communication and Technology, Guru Gobind Singh Indraprastha University
- PhD Scholar, Ankita, University School of Information, Communication and Technology, Guru Gobind Singh Indraprastha University
- Bhavdeep Singh Nijhawan
