# Benchmarking SVM and CNN Architectures for Medical Image Classification and Transfer Learning

## Abstract
This report details the implementation, methodology, and comparative benchmarking of automated diagnostic architectures—specifically analyzing the scaling resilience of a Linear Support Vector Machine (SVM) against a 2D Convolutional Neural Network (CNN). The study is divided into three analytical phases: baseline lung radiography classification, algorithmic performance under data starvation, and the efficiency of cross-domain Transfer Learning (migrating feature-detection from the lungs to the brain).

### Dataset Resources
- **Chest X-Ray (Pneumonia) Dataset:** Derived from Kaggle Medical Image Classification repositories.
- **Brain Tumor MRI Dataset:** 4-Class diagnostic set (Glioma, Meningioma, Pituitary, No Tumor).

---

## 1. Baseline Architecture & Pathology Diagnostics
The primary objective was establishing an automated deep learning baseline capable of detecting pneumonia via raw chest radiography.

**Methodology:**
Images were ingested, restricted to grayscale to eliminate irrelevant color variance, and mathematically normalized to a 0-1 scale. A deep architecture was constructed utilizing two sequential `Conv2D` layers to map raw pixels into spatial feature data. This required heavy downsampling via `MaxPooling2D` to prevent the network from memorizing the specific hospital environments (overfitting) rather than learning the pathology.

**Evaluation & Outcomes:**
Because simple 'Accuracy' is a dangerously flawed metric in medical diagnostics (a model could blindly guess "Healthy" on every image and score highly on unbalanced data), the CNN was evaluated across four strict criteria:
- **Accuracy**
- **F1-Score** (The harmonic mean of precision and recall)
- **Sensitivity** (True Positive Rate: crucial for preventing un-diagnosed patients)
- **Specificity** (True Negative Rate)

When fed 100% of the data, the deep convolutional structure successfully maps edge-abnormalities, demonstrating a high classification viability for thoracic infections.

---

## 2. Resilience to Data Starvation (SVM vs CNN)
To determine exactly how classical Machine Learning limits compare to Deep Learning scaling, a synthetic empirical data-starvation environment was induced. The core training data was severely partitioned into looping splits of 20%, 40%, 60%, 80%, and full 100% datasets.

**Methodology:**
At every single data boundary, two identical training pipelines were launched:
1. The 2D medical images were flattened into 1D numerical arrays and passed into a classical `SVC` (Support Vector Classifier) utilizing a Linear kernel.
2. The exact same images were kept in 2D grids and fed into an auxiliary CNN.

**Outcomes & Technical Thinking:**
By plotting the trajectory of both architectures from 20% to 100%, a definitive trend emerges. The classical Linear SVM struggles to draw mathematical hyperplanes through raw imagery; its accuracy and F1-Scores plateau early, regardless of how much extra data is provided.

Conversely, the CNN thrives on pure data density. While it struggles at the 20% mark, the introduction of the 60%, 80%, and ultimately 100% datasets allows its deep convolutional filters to mathematically compound geometry. The outcome proves that Deep Learning architectures strictly require high-volume data lakes to outperform classical algorithms via complex edge detection.

---

## 3. Cross-Domain Transfer Learning (Lungs to Brain)
The final hypothesis tested the true capability of neural feature-maps: Can an AI trained exclusively to detect ribs and lung infections accelerate diagnostics for Brain Tumors?

**Methodology:**
To answer this, the proven Chest X-Ray CNN was stripped of its pneumonia binary-output logic. Its core vision layers were locked and permanently frozen into the memory block (representing 18,816 non-trainable, preserved parameters). A brand-new deep dense classification head was mathematically grafted onto the frozen layer. To facilitate learning the 4 distinct brain pathologies, TensorFlow was forced to allocate over 1.6 Million new Trainable Parameters purely for the output connections.

For scientific validity, an identical "Opponent" CNN was constructed completely from scratch, carrying zero prior knowledge of human anatomy, utilizing randomized weight initialization.

**Outcomes & Technical Thinking:**
Both the Transfer Model and the Scratch Model were forcefully trained against the exact same explicit Brain Tumor directory for 10 sequential Epochs. 

Upon visualizing the validation accuracy trends side-by-side, the Transfer Model achieves geometric convergence drastically faster and maintains a systematically higher accuracy ceiling. This confirms a crucial medical imaging hypothesis: **Convolutional filters trained to map the geometric edges of human ribs and lung infiltrates universally translate to mapping tumor perimeters in the human brain**. Transfer Learning effectively bypasses the immense computational cost of teaching a network how to "see", safely repurposing that knowledge into an entirely different biological domain.
