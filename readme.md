# Real-Time multiclass Image Classification using CNN with Tensorflow and Flask


## Overview
Convolutional Neural Networks (CNNs) are powerful tools for working with images and videos. They automatically extract useful information from images to classify them accurately. This project provides a brief introduction to CNN and related concepts, and it builds a CNN model for image classification, which is tested for real-time prediction.

---

## Aim
- Build Convolutional Neural Network model to classify images into different classes.
- Deploy the model using Flask.

---

## Why Use CNN?
Before CNNs, image classification was a labor-intensive task, as manual feature engineering was required. CNNs use automatic feature extraction through convolution to create feature maps containing valuable image information, which is used for classification.

---

### When to Use CNN?
- Image Classification
- Image Segmentation
- Video Analysis
- Object Detection

---

## Tech Stack
- Language: `Python`
- Libraries: `TensorFlow`, `Pandas`, `Matplotlib`, `Keras`, `Flask`, `Pathlib`

---

## Data Description
The dataset used in this project contains images of driving licenses, social security cards, and other categories. The images have various shapes and sizes and are preprocessed before modeling.

---

## Approach
1. Data Loading
2. Data Preprocessing
3. Model Building and Training
4. Data Augmentation
5. Deployment

---

## Modular Codes Overview

1. `input`: Contains training and testing data folders, each further divided into driving license, social security, and others.
2. `output`: Contains testing images and the `cnn-model.h5` file (saved model after training).
3. `src`: The core of the project, housing modularized code for all the steps, including:
   - `ML_pipeline`: A folder with functions organized into different Python files.
   - `Engine.py`: This file calls the Python functions defined in `ML_pipeline`.

---

## Getting Started

1. Create virtual environment and install all the required libraries from requirements.txt 
      `Command: pip install -r requirements.txt`

2.  Make sure you are using python version between 3.5 to 3.8. Tensorflow doesn't work beyond 3.8.

3. In Deployement notebook (Model_API.ipynb) make sure you enter the correct url.

4. To check host and URL 
    - Run the deploy file and and at last of the terminal you will see the address of URL running on `http://192.168.29.219:5001/`

---

