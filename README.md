# cnn_hotodg_classifier

A binary image classification project built using **TensorFlow and Keras**, inspired by the famous *Not Hotdog* app from Silicon Valley.

The project explores how Convolutional Neural Networks (CNNs) can learn to distinguish hotdogs from other food items using image data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Navoditdev/cnn_hotodg_classifier/blob/main/hotdog_cnn.ipynb)

## 📌 Project Overview

This project implements a CNN trained from scratch on the **Food-101 dataset**.

The original dataset contains 101 food categories. For this project, the classification problem was converted into a binary task:

- **Hotdog (1):** Images belonging to the hotdog category.
- **Not Hotdog (0):** Images belonging to all other food categories.

The model learns visual features from images and produces a classification score indicating whether an image contains a hotdog.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Dataset:** TensorFlow Datasets (TFDS)
- **Libraries:** NumPy, Pandas, Matplotlib
- **Environment:** Google Colab

## 📊 Dataset & Preprocessing

The project uses the [Food-101 dataset](https://www.tensorflow.org/datasets/catalog/food101).

Preprocessing steps include:

1. Resizing images to **128 × 128 pixels**.
2. Converting the original 101-class labels into binary labels.
3. Using weighted sampling to address class imbalance.
4. Normalizing pixel values to the range [0, 1].
5. Applying data augmentation during training.

### Data Augmentation

To improve generalization, the model uses:

- Random horizontal flipping
- Random rotation
- Dropout regularization

The training pipeline also uses TensorFlow's `tf.data` API with batching, caching, and prefetching.

## 🧠 CNN Architecture

The model consists of three convolutional layers followed by fully connected layers.

```text
Input Image (128 × 128 × 3)
          |
     Rescaling
          |
   Data Augmentation
          |
   Conv2D (128, 3×3, ReLU)
          |
   MaxPooling2D (2×2)
          |
     Dropout (0.25)
          |
   Conv2D (64, 3×3, ReLU)
          |
   MaxPooling2D (2×2)
          |
     Dropout (0.25)
          |
   Conv2D (32, 3×3, ReLU)
          |
        Flatten
          |
   Dense (128, ReLU)
          |
     Dropout (0.25)
          |
      Dense (1)
          |
     Output Logit
```

The model was trained **from scratch**, without using pretrained CNN weights.

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Cross-Entropy (from logits) |
| Epochs | 50 |
| Batch Size | 16 |
| Framework | TensorFlow / Keras |

## 📈 Results

The following metrics were recorded in the final training epoch:

| Metric | Value |
|--------|-------|
| Training Accuracy | 84.65% |
| Validation Accuracy | 74.65% |
| Training Loss | 0.312 |
| Validation Loss | 0.643 |

The model demonstrated the ability to learn distinguishing features between hotdog and non-hotdog images.

However, the difference between training and validation performance suggests opportunities to improve generalization.

**Evaluation note:** These are metrics reported in the notebook, not independently verified test-set results. The validation pipeline repeats positive examples, and the default accuracy threshold may not be appropriate for the model's raw-logit output.

## 🚀 Running the Project

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top.
2. Enable GPU acceleration if available.
3. Run the notebook cells sequentially.
4. The Food-101 dataset will be downloaded automatically.
5. Train the model and inspect the results.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/Navoditdev/cnn_hotodg_classifier.git
cd cnn_hotodg_classifier
```

Install the dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib jupyter
```

Launch Jupyter:

```bash
jupyter notebook hotdog_cnn.ipynb
```

Note: Training requires downloading Food-101 and may be computationally intensive.

## 🔬 Limitations & Future Improvements

Potential improvements include:

- Evaluating the model on a separate, unseen test set.
- Correcting the accuracy metric threshold for raw logits.
- Improving class balancing without repeating validation examples.
- Experimenting with pretrained architectures such as ResNet or MobileNet.
- Implementing early stopping and hyperparameter tuning.
- Analyzing misclassified images using a confusion matrix.
- Building a simple web interface for image predictions.

## 📁 Repository Structure

```text
cnn_hotodg_classifier/
├── hotdog_cnn.ipynb    # CNN implementation and training
└── README.md           # Project documentation
```

## 🎯 What I Learned

Through this project, I explored:

- Designing and training convolutional neural networks.
- Image preprocessing and data augmentation.
- Working with TensorFlow's data pipelines.
- Addressing class imbalance in classification tasks.
- Using binary cross-entropy and gradient-based optimization.
- Evaluating model performance and identifying generalization issues.

---

**Author:** [Navoditdev](https://github.com/Navoditdev)

*An introductory deep learning project exploring computer vision and binary image classification.*
