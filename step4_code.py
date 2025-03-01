{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOGBdHTxodrfYATMwFaZYcK",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/BhavdeepSinghNijhawan/Machine-Learning-for-Enhanced-Cybersecurity-Solutions/blob/main/step4_code.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hj5jCNWJ34YW",
        "outputId": "daab44a9-d622-4ca2-978f-e0855b3f06f7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.11/dist-packages (2.18.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (1.26.4)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.2.10)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from tensorflow) (24.2)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.25.6)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.32.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from tensorflow) (75.1.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.5.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.12.2)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.2)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.70.0)\n",
            "Requirement already satisfied: tensorboard<2.19,>=2.18 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.18.0)\n",
            "Requirement already satisfied: keras>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.8.0)\n",
            "Requirement already satisfied: h5py>=3.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.12.1)\n",
            "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.4.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.37.1)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.0.8)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.14.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2025.1.31)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.7)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.1.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (2.18.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)\n"
          ]
        }
      ],
      "source": [
        "!pip install tensorflow numpy"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Input, Dense\n",
        "from tensorflow.keras.models import Model"
      ],
      "metadata": {
        "id": "kmZoRMw64GKs"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define input layer\n",
        "input_layer = Input(shape=(256,))\n",
        "encoded = Dense(128, activation='relu')(input_layer)\n",
        "decoded = Dense(256, activation='sigmoid')(encoded)\n",
        "\n",
        "# Create and compile autoencoder model\n",
        "autoencoder = Model(input_layer, decoded)\n",
        "autoencoder.compile(optimizer='adam', loss='mse')\n"
      ],
      "metadata": {
        "id": "zmWGaTwx4L5S"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate random training keys (assuming binary-like data)\n",
        "np.random.seed(42)\n",
        "X_train = np.random.randint(0, 2, size=(10000, 256))  # 10,000 samples\n",
        "\n",
        "# Introduce noise\n",
        "X_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)\n",
        "X_noisy = np.clip(X_noisy, 0, 1)  # Ensure values remain between 0 and 1\n"
      ],
      "metadata": {
        "id": "mhajE_Qn4PfY"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "autoencoder.fit(X_noisy, X_train, epochs=10, batch_size=32, verbose=1)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jDtH8_Uh4Q5u",
        "outputId": "43087363-1b97-428d-e3b3-be0ba55c1ab6"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 3ms/step - loss: 0.2422\n",
            "Epoch 2/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - loss: 0.1818\n",
            "Epoch 3/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 4ms/step - loss: 0.1495\n",
            "Epoch 4/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 2ms/step - loss: 0.1331\n",
            "Epoch 5/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - loss: 0.1235\n",
            "Epoch 6/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - loss: 0.1186\n",
            "Epoch 7/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - loss: 0.1165\n",
            "Epoch 8/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - loss: 0.1150\n",
            "Epoch 9/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - loss: 0.1137\n",
            "Epoch 10/10\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - loss: 0.1128\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.history.History at 0x7b96757d7cd0>"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate a sample chaotic key with noise\n",
        "chaotic_key = np.random.randint(0, 2, size=(256,))\n",
        "noisy_chaotic_key = chaotic_key + np.random.normal(0, 0.1, chaotic_key.shape)\n",
        "noisy_chaotic_key = np.clip(noisy_chaotic_key, 0, 1)\n",
        "\n",
        "# Use autoencoder for correction\n",
        "corrected_key = autoencoder.predict(np.array([noisy_chaotic_key]))\n",
        "corrected_key = np.round(corrected_key)  # Convert back to binary-like format\n",
        "\n",
        "print(\"Original Key:   \", chaotic_key)\n",
        "print(\"Noisy Key:      \", noisy_chaotic_key.round(2))\n",
        "print(\"Corrected Key:  \", corrected_key.astype(int))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vHpGPSEE4X9h",
        "outputId": "18b22e21-a197-4d78-fb1e-dae352c1ceba"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 69ms/step\n",
            "Original Key:    [0 1 0 0 0 1 1 1 0 1 1 0 1 0 1 0 0 1 1 1 0 0 0 1 1 0 1 1 1 1 1 0 0 1 0 0 0\n",
            " 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0 1 1 1 0 1 0 0 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0\n",
            " 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 1 1 1 1 0 1 0\n",
            " 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 1 1 1 0 0 0 0 1 0 1 1 0 1 0 0\n",
            " 1 1 1 0 1 1 1 1 0 0 1 0 1 0 1 0 1 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 1\n",
            " 1 1 1 0 0 0 0 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0 0 0 1 1 0 1 0 1 1 0 0 0\n",
            " 0 1 1 0 1 1 1 0 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 1]\n",
            "Noisy Key:       [0.02 0.9  0.13 0.   0.   1.   1.   0.97 0.13 0.97 1.   0.09 1.   0.\n",
            " 0.82 0.1  0.01 0.96 1.   1.   0.01 0.   0.02 0.99 1.   0.   0.87 0.84\n",
            " 1.   1.   0.94 0.02 0.21 1.   0.   0.01 0.11 0.11 1.   0.14 0.93 1.\n",
            " 1.   0.12 1.   0.   0.   0.09 0.95 0.06 0.   0.08 0.97 1.   0.93 0.\n",
            " 0.93 0.   0.   1.   0.95 0.   0.96 1.   1.   0.95 1.   0.09 0.96 0.08\n",
            " 0.   0.   0.1  0.   1.   1.   0.   0.   0.16 0.89 0.12 0.   0.   1.\n",
            " 0.14 0.08 0.98 0.98 0.83 1.   1.   0.98 0.1  0.02 0.   0.06 1.   0.\n",
            " 0.96 1.   0.9  0.   0.   1.   0.94 0.95 0.97 1.   0.   0.83 0.03 0.15\n",
            " 1.   1.   0.   1.   0.89 0.83 0.06 1.   0.   1.   0.   1.   1.   0.9\n",
            " 0.   0.99 1.   1.   0.   0.   1.   0.95 1.   0.83 0.04 0.   0.   0.\n",
            " 0.91 0.13 1.   0.77 0.   0.84 0.   0.12 1.   0.97 0.96 0.   0.89 0.84\n",
            " 0.9  1.   0.04 0.   0.97 0.   1.   0.09 0.94 0.   0.97 0.   1.   1.\n",
            " 1.   1.   0.97 0.   0.93 0.98 1.   0.05 0.03 1.   0.99 0.95 0.11 0.\n",
            " 0.99 0.91 0.96 0.94 1.   0.97 0.   0.08 0.09 0.   0.   0.98 0.13 0.04\n",
            " 0.94 0.81 0.   1.   1.   0.95 0.01 1.   1.   0.   1.   1.   1.   0.\n",
            " 0.   0.01 1.   1.   0.   1.   0.06 1.   1.   0.02 0.   0.02 0.09 0.95\n",
            " 1.   0.01 1.   1.   1.   0.   0.14 0.17 1.   0.   0.06 1.   0.   0.\n",
            " 0.97 0.98 0.98 0.   0.86 0.   0.13 0.96 0.95 0.02 1.   0.   1.   0.99\n",
            " 0.04 1.   1.   0.95]\n",
            "Corrected Key:   [[0 1 0 1 1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 0 1 0 0 0 1 1 1 1 0 1 1 0 1\n",
            "  0 1 0 0 1 1 0 0 1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 0 1 1 0 1 0 1 0 0 0\n",
            "  0 0 1 0 0 1 0 0 0 0 0 1 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 1 0 0\n",
            "  0 1 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 0 0 1 0 1 1\n",
            "  0 1 0 0 0 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 0 1 1 1\n",
            "  0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 0 1 1 0 1 1 1 0 0 0 0 1 0 0\n",
            "  0 0 0 1 1 0 0 1 1 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 1 1 1 1 1 1 1\n",
            "  0 1 0 1]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4jD597zQ41Un",
        "outputId": "ad087ab4-317c-43b3-ed89-0e12f7741f9f"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "code = \"\"\"\n",
        "# Your step4_code.py content here\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Input, Dense\n",
        "from tensorflow.keras.models import Model\n",
        "\n",
        "# Define the autoencoder\n",
        "input_layer = Input(shape=(256,))\n",
        "encoded = Dense(128, activation='relu')(input_layer)\n",
        "decoded = Dense(256, activation='sigmoid')(encoded)\n",
        "autoencoder = Model(input_layer, decoded)\n",
        "autoencoder.compile(optimizer='adam', loss='mse')\n",
        "\n",
        "# Generate training data\n",
        "np.random.seed(42)\n",
        "X_train = np.random.randint(0, 2, size=(10000, 256))\n",
        "X_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)\n",
        "X_noisy = np.clip(X_noisy, 0, 1)\n",
        "\n",
        "# Train the autoencoder\n",
        "autoencoder.fit(X_noisy, X_train, epochs=50, batch_size=32, verbose=1)\n",
        "\n",
        "# Test on a noisy key\n",
        "chaotic_key = np.random.randint(0, 2, size=(256,))\n",
        "noisy_chaotic_key = chaotic_key + np.random.normal(0, 0.1, chaotic_key.shape)\n",
        "noisy_chaotic_key = np.clip(noisy_chaotic_key, 0, 1)\n",
        "\n",
        "corrected_key = autoencoder.predict(np.array([noisy_chaotic_key]))\n",
        "corrected_key = np.round(corrected_key)\n",
        "\n",
        "print(\"Original Key:   \", chaotic_key)\n",
        "print(\"Noisy Key:      \", noisy_chaotic_key.round(2))\n",
        "print(\"Corrected Key:  \", corrected_key.astype(int))\n",
        "\"\"\"\n",
        "\n",
        "with open(\"/content/step4_code.py\", \"w\") as f:\n",
        "    f.write(code)\n"
      ],
      "metadata": {
        "id": "z710Mr3H5JvR"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/BhavdeepSinghNijhawan/Machine-Learning-for-Enhanced-Cybersecurity-Solutions.git\n",
        "!mv /content/step4_code.py /content/Machine-Learning-for-Enhanced-Cybersecurity-Solutions/\n",
        "%cd /content/Machine-Learning-for-Enhanced-Cybersecurity-Solutions/\n",
        "\n",
        "!git config --global user.email \"bhavdeepnijhawan22@gmail.com\"  # Replace with your GitHub email\n",
        "!git config --global user.name \"BhavdeepSinghNijhawan\"\n",
        "\n",
        "!git add step4_code.py\n",
        "!git commit -m \"Added Step 4: ML-powered Error Correction using Autoencoder\"\n",
        "!git push https://github.com/BhavdeepSinghNijhawan/Machine-Learning-for-Enhanced-Cybersecurity-Solutions.git main\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1nZXRcit5AfY",
        "outputId": "f295d05d-fbb0-4a0c-e5c0-ef4c60c9a202"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "fatal: destination path 'Machine-Learning-for-Enhanced-Cybersecurity-Solutions' already exists and is not an empty directory.\n",
            "/content/Machine-Learning-for-Enhanced-Cybersecurity-Solutions\n",
            "[main a03073c] Added Step 4: ML-powered Error Correction using Autoencoder\n",
            " 1 file changed, 34 insertions(+)\n",
            " create mode 100644 step4_code.py\n",
            "fatal: could not read Username for 'https://github.com': No such device or address\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!apt install gh  # Install GitHub CLI\n",
        "!gh auth login   # Follow prompts to log in\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9r-WEXAl5Vqx",
        "outputId": "6c72d3b8-a7f6-4c32-a7d2-d4303eabef4d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "The following NEW packages will be installed:\n",
            "  gh\n",
            "0 upgraded, 1 newly installed, 0 to remove and 29 not upgraded.\n",
            "Need to get 6,242 kB of archives.\n",
            "After this operation, 33.7 MB of additional disk space will be used.\n",
            "Get:1 http://archive.ubuntu.com/ubuntu jammy/universe amd64 gh amd64 2.4.0+dfsg1-2 [6,242 kB]\n",
            "Fetched 6,242 kB in 2s (3,775 kB/s)\n",
            "Selecting previously unselected package gh.\n",
            "(Reading database ... 124947 files and directories currently installed.)\n",
            "Preparing to unpack .../gh_2.4.0+dfsg1-2_amd64.deb ...\n",
            "Unpacking gh (2.4.0+dfsg1-2) ...\n",
            "Setting up gh (2.4.0+dfsg1-2) ...\n",
            "Processing triggers for man-db (2.10.2-1) ...\n",
            "\u001b7\u001b[?25l\u001b8\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into?\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? G\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? Gi\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? Git\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitH\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHu\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHub\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b[0;39m  GitHub Enterprise Server\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHub.\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHub.c\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHub.co\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into? GitHub.com\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> GitHub.com\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[?25h\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat account do you want to log into?\u001b[0m\u001b[0;36m GitHub.com\u001b[0m\n",
            "\u001b7\u001b[?25l\u001b8\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations?\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b[0;39m  SSH\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations? H\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b[0;39m  SSH\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations? HT\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations? HTT\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations? HTTP\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations? HTTPS\u001b[0m  \u001b[0;36m[Use arrows to move, type to filter]\u001b[0m\n",
            "\u001b[0;1;36m> HTTPS\u001b[0m\n",
            "\u001b7\u001b[1A\u001b[0G\u001b8\u001b[?25h\u001b8\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[1A\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mWhat is your preferred protocol for Git operations?\u001b[0m\u001b[0;36m HTTPS\u001b[0m\n",
            "\u001b[0G\u001b[2K\u001b[0;1;92m? \u001b[0m\u001b[0;1;99mAuthenticate Git with your GitHub credentials? \u001b[0m\u001b[0;39m(Y/n) \u001b[0m\u001b[?25l\u001b7\u001b[999;999f\u001b[6n"
          ]
        }
      ]
    }
  ]
}