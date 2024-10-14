# brain-tumor-mri-classifier

brain_tumor_classification/
│
├── app.py                # Main Streamlit application
├── model.py              # Model training and evaluation script
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── models/               # Directory to store trained models
    ├── cnn_model.h5     # CNN model file
    └── vgg_model.h5      # Transfer learning model file
    

Overview

This project focuses on the classification of brain tumors using MRI images with deep learning models. The objective is to accurately detect and classify brain tumors into specific categories to assist in medical diagnosis. The project includes two methods:

Custom Convolutional Neural Network (CNN).
Transfer Learning using VGG16 pre-trained on ImageNet.
The trained models are integrated into a Streamlit web application, allowing users to upload MRI images and get real-time classification predictions with confidence scores.

Motivation

Brain tumors are life-threatening conditions, and early detection is critical for effective treatment. MRI (Magnetic Resonance Imaging) is a widely used method for detecting abnormalities in the brain. This project aims to leverage the power of deep learning to classify brain tumors with high accuracy, which could aid in automating the diagnostic process and assist radiologists and doctors.

Dataset

The models were trained on a dataset of brain MRI images, which includes the following categories:

Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor
The dataset was preprocessed and augmented to improve generalization and prevent overfitting during model training.

Image Size: All images were resized to 256x256 pixels.
Normalization: Pixel values were normalized to a range of 0-1.

Model Architectures

1. Custom CNN
The custom CNN was designed from scratch to classify MRI images. It consists of several convolutional layers followed by max-pooling, and dense layers. The architecture details are as follows:

Convolutional Layers: Extract features from the input images using filters.
Max Pooling: Reduce the spatial dimensions of the feature maps.
Dense Layers: Fully connected layers for classification.
Activation: ReLU was used for non-linearity in hidden layers, and softmax was used for output classification.
Optimizer: Adam optimizer was used to speed up convergence.
Loss Function: Sparse Categorical cross-entropy.

2. Transfer Learning with VGG16
VGG16 is a popular deep learning model pre-trained on ImageNet. We utilized this architecture to leverage its pre-trained weights and fine-tuned the model for our task.

Base Model: VGG16 with ImageNet weights (frozen for feature extraction).
Added Layers:
Flattening the output of the VGG16.
Adding fully connected dense layers for classification.
Final softmax layer for output.

Why Transfer Learning?

Transfer learning allows us to take advantage of a powerful pre-trained model and fine-tune it for our specific classification task. This drastically reduces the amount of training time while maintaining high accuracy.

Training Strategy

Batch Size: A batch size of 32 was used for training both models.
Epochs: The models were trained for up to 10 epochs.
Early Stopping: Early stopping was implemented to prevent overfitting. The training stops if the validation loss does not improve for 3 consecutive epochs.
Validation Split: A validation split of 20% was used during training to evaluate model performance.

Results
The model achieved high accuracy on the test set with the following performance:

Custom CNN Accuracy: ~94%
Transfer Learning (VGG16) Accuracy: ~95%

Streamlit App
The trained model is deployed using a Streamlit app. It allows users to upload an MRI image of the brain and returns the predicted tumor category along with the confidence level.

Key Features:

File Upload: Upload MRI images in JPG or PNG format.
Real-Time Classification: Get real-time predictions and confidence scores.
Easy-to-Use Interface: The Streamlit app provides an intuitive interface for interacting with the model.

Future Improvements

Data Augmentation: Implement advanced data augmentation techniques to further improve model generalization.
Hyperparameter Tuning: Perform hyperparameter tuning to achieve better performance.
Increase Dataset Size: More MRI images could improve the model’s robustness.
Additional Tumor Types: Extend the model to classify additional types of brain tumors.
Conclusion
This project demonstrates how deep learning models, particularly CNNs and transfer learning, can be effectively used for medical image classification. With real-time predictions in a user-friendly Streamlit app, it provides a practical application that could assist medical professionals in diagnosing brain tumors.
