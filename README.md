🍅 Tomato Disease Prediction
A deep learning-based web application that identifies common tomato leaf diseases from images. Built using Convolutional Neural Networks (CNN) and a custom dataset, this project aims to assist farmers and researchers in early disease detection.​

 Features:
Upload tomato leaf images to detect diseases.​

Real-time predictions using a trained CNN model.​

User-friendly web interface built with Flask.​

Supports detection of multiple tomato diseases.​

Model Architecture:
The CNN model consists of:​
Convolutional layers with ReLU activation​
MaxPooling layers​
Fully connected Dense layers​
Softmax output layer for multi-class classification

Project Architechture:
tomato-disease-prediction/
├── static/
│   ├── css/
│   ├── image/
│   ├── js/
│   └── models/
│       └── uploads/ (contains images or model files)
├── templates/
│   ├── chatbot.html
│   ├── disease_prediction.html
│   ├── home.html
│   ├── login.html
│   └── signin.html
├── app.py



