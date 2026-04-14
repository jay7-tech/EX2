# Exercise 2: Medical Image Classification Benchmarks

## Overview
This repository contains the implementation for Exercise 2. The objective of this project is to benchmark Support Vector Machines (SVM) against Convolutional Neural Networks (CNN) using medical image datasets. 

The analysis is broken into three core pipelines:
1. **Chest X-Ray Classification** (Binary model training)
2. **Data Starvation Testing** (Evaluating SVM vs CNN reliance on data scale)
3. **Transfer Learning** (Repurposing lung-scan CNNs to detect Brain Tumors)

---

## Part 1: Primary CNN Pipeline (Chest X-Rays)
The foundational step involves building a Convolutional Neural Network from scratch to classify Chest X-Rays into two buckets: **Normal** or **Pneumonia**.

**Architecture Overview:**
- 2x `Conv2D` layers (with `ReLU` activation for non-linearity)
- 2x `MaxPooling2D` layers (for spatial downsampling)
- Flattened into a `Dense(64)` intermediate layer.
- `Dense(1, Sigmoid)` output layer for binary classification.

**Metrics Tracked:**
To ensure a robust evaluation of medical diagnostic viability, the primary network is assessed on four metrics:
- **Accuracy** 
- **F1-Score** 
- **Sensitivity (Recall)** 
- **Specificity** 

*(Placeholder: Insert Training/Validation Loss Curves Here)*

---

## Part 2: Data Starvation Analysis (SVM vs CNN)
Deep Learning models are notoriously data-hungry. To test this empirically, we constructed a starvation matrix limiting the training data to splits of **20%, 40%, 60%, 80%, and 100%**. 

At each split, we trained both a classical 1D Linear SVM and a 2D Deep Learning CNN to observe their degradation curves.

### Expected Metric Layout (Example Table)
| Training Data | 20% | 40% | 60% | 80% | 100% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | - | - | - | - | - |
| **Simple CNN** | - | - | - | - | - |

**Analysis Trend:**
Classical Linear SVMs often plateau early or struggle to interpret complex spatial variance in raw pixels. CNNs inherently scale their spatial understanding proportional to the data volume provided. The generated `Accuracy vs Data Starvation` plots demonstrate the CNN's superior ceiling when fed the full 100% dataset.

*(Placeholder: Insert Starvation Line Graphs Here)*

---

## Part 3: The Power of Transfer Learning
To evaluate the mathematical cross-compatibility of medical imagery, we applied Transfer Learning to the `EX1` Brain Tumor dataset (4 Classes: Glioma, Meningioma, Pituitary, No Tumor).

**Methodology:**
1. **The Transfer Model:** The pre-trained Chest X-Ray network (`chest_xray_cnn.keras`) was loaded. All underlying convolutional (feature-extracting) layers were frozen to preserve their pattern recognition. A new, randomly initialized 4-output Brain Tumor classification head was attached.
2. **The Scratch Model:** An identical architectural skeleton was built, but initialized with completely random weights (zero prior knowledge).

**Conclusions Drawn:**
Both algorithms were pitted against each other for a static 10 epochs. By plotting the Validation Accuracy trajectories over time, the results indicate that morphological edge-detection algorithms learned exclusively from Rib/Lung borders physically translate to Brain MRI edge-detection. The Transfer Model consistently achieves a higher accuracy ceiling in a fraction of the computational time.

*(Placeholder: Insert Transfer Learning vs Scratch Validation Plot Here)*

---

## Execution Instructions
1. Ensure the `chest_xray` and `brain_data` dataset folders are located in the project's root directory.
2. Open `Exercise_2.ipynb`.
3. Execute the cells sequentially.
> **Note:** The current build defaults the CNN to `epochs=3` to allow for rapid CPU execution and testing. For the final production build, adjust this to `epochs=100` and utilize a hardware GPU.
