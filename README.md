# GESTATIONAL-AGE-PREDICTION-FROM-FETAL-MRI-USING-DEEP-LEARNING
This project presents a deep learning-based approach for predicting fetal gestational age from multi-planar MRI. The model uses a VGG16 CNN backbone enhanced with an attention module and processes axial, coronal, and sagittal MRI slices simultaneously to learn anatomical patterns linked to brain maturation.

**📌 Objective**
Build a regression model that predicts gestational age (in days) from multi-planar fetal MRI using CNN feature extraction, attention weighting, and multi-view fusion.

**🧠 Methodology**

**Model Architecture**

**CNN Backbone: VGG16 (pretrained on ImageNet)**

Used to extract spatial features from each MRI plane.

**Attention Module**
1×1 convolution → sigmoid activation → element-wise multiplication
Highlights anatomically relevant regions.

**Multi-Planar Fusion**
Features from axial, coronal, and sagittal views are flattened and concatenated.

**Regression Head**
256-unit Dense layer → Dropout(0.5) → Linear output (gestational age in days)

**Training Workflow**

**Step 1:** Load MRI slices from axial, coronal, and sagittal folders.

**Step 2:** Resize images to 224×224 and normalize pixel values to [0,1].

**Step 3:** Stratified 5-fold cross-validation based on gestational age grouping.

**Step 4:** Build a fresh VGG16-attention model for each fold.

**Step 5**: Fine-tune selected VGG16 layers (last 100 layers trainable).

**Step 6:** Train with Huber loss and Adam optimizer.

**Step 7:** Evaluate predictions using MAE and R².

**🎯 Dataset**

The dataset contains 261 fetal T2-weighted MRI scans, covering gestational ages from 19 to 39 weeks (median: 29 weeks). Gestational age was determined using the Last Menstrual Period (LMP), providing consistent clinical labels.

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

**Metrics:**

Mean Absolute Error (MAE)

R² Score

**📊 Results**

Using VGG16 + Multi-Planar Fusion + Attention, the model achieved:

MAE ≈ 4.5 days

R² ≈ 0.95

Performance was stable across all folds with low variance.

**🔍 Interpretability**

Attention heatmaps were generated for:

Axial

Coronal

Sagittal

These maps highlight spatial regions that contributed most to gestational-age prediction.

**🛠 Technologies**

TensorFlow / Keras

NumPy, Pandas, scikit-learn

OpenCV

Python (Anaconda + Spyder)

GPU-accelerated training environment
