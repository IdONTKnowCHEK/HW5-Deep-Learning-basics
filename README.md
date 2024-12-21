

# **Deep Learning Projects Overview**

This repository contains three distinct machine learning projects that demonstrate classification tasks using different datasets and neural network architectures.

## GPT History
[GPT](https://chatgpt.com/share/6766fd07-61cc-800c-899e-8e320e5ff838)

---

## **1. Iris Classification**

### **Overview**
This project involves classifying the Iris dataset using:
- TensorFlow Keras
- PyTorch
- PyTorch Lightning

### **Dataset**
The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) consists of 150 samples with four features (sepal length, sepal width, petal length, petal width) and three classes.

### **Models**
- **TensorFlow Keras**: A simple feed-forward neural network is implemented with three dense layers and softmax activation for output.
- **PyTorch**: The model is implemented using PyTorch’s nn module, with training and validation loops manually coded.
- **PyTorch Lightning**: Uses the PyTorch Lightning framework for a structured training pipeline.

### TensorBoard
#### TensorFlow Keras
![image](https://github.com/user-attachments/assets/62c51002-e153-4ffb-92e5-3f514c13ccda)
#### Pytorch
![image](https://github.com/user-attachments/assets/8a93c171-457d-49a6-a711-859f53806bf3)

#### PyTorch Lightning
![image](https://github.com/user-attachments/assets/fc9a1bce-4ec0-44f6-874c-b7b4a4ae2f88)


### **Results**
Achieves an accuracy of over 95% on the Iris dataset with minimal hyperparameter tuning.

---

## **2. Handwriting Recognition**

### **Overview**
This project implements handwriting recognition on the MNIST dataset using:
- Dense Neural Networks (DNN)
- Convolutional Neural Networks (CNN)

### **Dataset**
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 70,000 grayscale images of handwritten digits (28x28 pixels) belonging to 10 classes (digits 0-9).

### **Models**
- **DNN**: Uses fully connected layers with ReLU activation and dropout for regularization.
- **CNN**: Incorporates convolutional and pooling layers to extract spatial features, followed by dense layers for classification.

### **Results**
The CNN model outperforms the DNN, achieving an accuracy of over 99% on the MNIST dataset.

---

## **3. CIFAR-10 Image Classification**

### **Overview**
This project leverages the VGG19 pre-trained model for image classification on the CIFAR-10 dataset.

### **Dataset**
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 color images (32x32 pixels) in 10 classes, with 50,000 for training and 10,000 for testing.

### **Model**
- **VGG19 Pre-trained Model**: The model uses ImageNet-pretrained weights for feature extraction, and custom dense layers are added for CIFAR-10 classification.

### **Results**
Achieves high accuracy (~90%) on the CIFAR-10 dataset with minimal training, thanks to transfer learning.

---

## **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/IdONTKnowCHEK/HW5-Deep-Learning-basics
   cd HW5-Deep-Learning-basics
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Notebooks**
   Open the respective Jupyter notebooks for each project and run the code:
   - Iris Classification: `iris_classification.ipynb`
   - Handwriting Recognition: `handwriting_recognition.ipynb` (DNN and CNN versions)
   - CIFAR-10 Classification: `cifar10_vgg19.ipynb`

---

## **Requirements**
- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x
- PyTorch Lightning
- Matplotlib
- Scikit-learn
- NumPy

---

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests to improve the projects.

---

## **License**
This project is licensed under the MIT License.
