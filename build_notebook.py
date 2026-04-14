import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    })

# --- CELL 1: Introduction ---
add_markdown("""# Exercise 2: The Plain English Guide
Hello! If you are confused by the assignment, let's break it down into very simple terms.

### What is the goal of this assignment?
In Exercise 1, you used a classic Machine Learning algorithm called an **SVM (Support Vector Machine)** to look at Brain Tumors. 

For Exercise 2, you are asked to do 4 things:
1. **New Dataset**: Do that exact same SVM process, but on a new set of images: **Chest X-Rays for Pneumonia**.
2. **Introduce Deep Learning**: Instead of just using the old SVM, build a modern **CNN (Convolutional Neural Network)** to look at the Chest X-Rays.
3. **The Data Starvation Test**: An SVM and a CNN learn very differently. The assignment asks: "What happens if we only give the algorithms 20% of the images? What about 40%? 80%?" You need to test both algorithms with less data and see which one breaks first.
4. **Transfer Learning**: Finally, take the CNN that just learned how to read Chest X-Rays, and force it to look at Brain Tumors. Because it already learned how to see "edges" and "shapes" in X-Rays, it will learn Brain Tumors much faster. This is called Transfer Learning.

That is it! We are simply comparing **SVMs** vs **CNNs** on Chest X-Rays, testing them with less data, and then recycling the CNN for Brain Tumors. 

Let's go step-by-step in these cells.""")

# --- CELL 2: Imports ---
add_markdown("""### Step 1: Tool Setup
First, we just import all the math and machinery we need, like TensorFlow (for the CNN) and Scikit-Learn (for the SVM).""")
add_code("""import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score""")

# --- CELL 3: Data Loading ---
add_markdown("""### Step 2: Loading the Chest X-Ray Data
We need to load the images. SVMs don't like heavy images, so we squish them down to 64x64 pixels and flatten them into a single line of numbers (1D array).""")
add_code("""# Pointing to the folder you downloaded
DATA_PATH = r"c:/Users/JAYADEEP GOWDA K B/Desktop/EXE/EX2/chest_xray"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TEST_DIR = os.path.join(DATA_PATH, "test")

def load_data_for_svm(directory):
    data = []
    labels = []
    # 0 = Normal, 1 = Pneumonia
    classes = {"NORMAL": 0, "PNEUMONIA": 1}
    
    for cls, label_val in classes.items():
        folder_path = os.path.join(directory, cls)
        if not os.path.exists(folder_path): continue
        
        # Load up to 500 images per class to keep your computer from crashing
        for i, img_file in enumerate(os.listdir(folder_path)):
            if img_file.startswith("."): continue
            if i > 500: break # Limiter
                
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten()) # Flatten into 1D for SVM
            labels.append(label_val)
            
    return np.array(data), np.array(labels)

print("Loading Data... (This takes a moment)")
X_train, y_train = load_data_for_svm(TRAIN_DIR)
X_test, y_test = load_data_for_svm(TEST_DIR)
print(f"Loaded {len(X_train)} training images.")""")

# --- CELL 4: PCA & SVM ---
add_markdown("""### Step 3: PCA + SVM Benchmark
Images have thousands of pixels. **PCA (Principal Component Analysis)** throws away the useless pixels (like the black background) and keeps only the most important features. 
Then, we test the SVM exactly as requested in the assignment (comparing `linear` vs `rbf` kernels).""")
add_code("""print("Applying PCA...")
pca = PCA(n_components=0.95) # Keep 95% of the important details
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("PCA finished.")

# We will test two types of SVMs
for kernel_type in ['linear', 'rbf']:
    print(f"Training SVM with {kernel_type} kernel...")
    svm = SVC(kernel=kernel_type)
    svm.fit(X_train_pca, y_train)
    
    accuracy = accuracy_score(y_test, svm.predict(X_test_pca))
    print(f"-> {kernel_type.upper()} Accuracy: {accuracy * 100:.2f}%")""")

# --- CELL 5: CNN Introduction ---
add_markdown("""### Step 4: The Deep Learning Solution (CNN)
Now we ditch the SVM entirely. A CNN is a neural network that uses "filters" to scan over the image and find spatial shapes (like a magnifying glass). It doesn't need PCA because it naturally learns what is important.

We will build the exact structure Kaggle recommends for this dataset.""")
add_code("""from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("Building the CNN...")
model = Sequential([
    # Layer 1: Finds simple edges
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    
    # Layer 2: Finds complex shapes
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Brains of the operation
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # Outputs 0 (Normal) or 1 (Pneumonia)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("CNN Architecture Ready!")""")

# --- CELL 6: What's next? ---
add_markdown("""### Step 5: What is left?
To finish the assignment as requested, we need to:
1. Run that CNN code we just built using only 20%, 40%, 60%, and 80% of the images and track the accuracy numbers.
2. Take that CNN model, freeze its layers, and retrain it on your Brain Tumor dataset from `EX1`.

**Take a moment to run the cells above.** If everything makes sense, we can add the final cells to do the data splitting loop and the Transfer Learning!""")


out_path = r"c:/Users/JAYADEEP GOWDA K B/Desktop/EXE/EX2/Exercise_2.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
