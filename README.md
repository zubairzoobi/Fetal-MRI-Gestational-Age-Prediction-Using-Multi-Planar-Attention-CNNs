# GESTATIONAL-AGE-PREDICTION-FROM-FETAL-MRI-USING-DEEP-LEARNING
This project presents a deep learning-based approach for predicting fetal gestational age from multi-planar MRI. The model uses a VGG16 CNN backbone enhanced with an attention module and processes axial, coronal, and sagittal MRI slices simultaneously to learn anatomical patterns linked to brain maturation.

**📌 Objective**
Build a regression model that predicts gestational age (in days) from multi-planar fetal MRI using CNN feature extraction, attention weighting, and multi-view fusion.

**🧠 Methodology**

**Model Architecture:** VGG16 (pretrained on ImageNet) Used to extract spatial features from each MRI plane.

**Attention Module:** 1×1 convolution → sigmoid activation → element-wise multiplication. Highlights anatomically relevant regions.

**Multi-Planar Fusion:** Features from axial, coronal, and sagittal views are flattened and concatenated.

**Regression Head:** 256-unit Dense layer → Dropout(0.5) → Linear output (gestational age in days)

**🧩 Training Workflow**

**Step 1:** Load axial, coronal, and sagittal MRI slices.

**Step 2:** Resize images to 224×224 and normalize pixel values to [0,1].

**Step 3:** Stratified 5-fold cross-validation based on gestational age grouping.

**Step 4:** Build a new VGG16-attention model for each fold.

**Step 5**: Fine-tune the last 100 VGG16 layers.

**Step 6:** Train using Huber loss + Adam.

**Step 7:** Evaluate using MAE and R².

**🎯 Dataset**

Dataset includes 261 fetal T2-weighted MRI scans ranging from 19–39 weeks gestation (median: 29 weeks). Gestational age labels derived from Last Menstrual Period (LMP).

Each subject includes multiple slices across three anatomical planes: Axial Coronal Sagittal

**⚙️ Training Details**

Optimizer: Adam (1e-4)

Loss: Huber

Batch Size: 8

Epochs: 100

Callbacks:

ReduceLROnPlateau

EarlyStopping (patience 10)

Cross-validation: 5 folds

Metrics: Mean Absolute Error (MAE), R² Score

**📊 Results**

Model performance with VGG16 + attention + multi-planar fusion:

MAE ≈ 4.5 days

R² ≈ 0.95

Performance was stable across all folds with low variance.

**🔍 Interpretability**

Attention heatmaps generated for axial, coronal, and sagittal views highlight spatial regions influencing the prediction output.

**🛠 Technologies**

Language: Python

Frameworks: TensorFlow , Keras, NumPy, Pandas, scikit-learn, OpenCV

Tools: Anaconda, Spyder, Git, GPU/ GPU-accelerated training environment
