import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import build_multiplanar_vgg16

def preprocess(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return img

def load_dataset(excel_path, dirs, use_planes):
    df = pd.read_excel(excel_path)
    X = {'axial': [], 'coronal': [], 'sagittal': []}
    y = []

    for _, row in df.iterrows():
        pid = str(row['patient_id'])

        if use_planes['axial']:
            ax_dir = os.path.join(dirs['axial'], pid)
            ax_files = sorted(glob.glob(ax_dir + "/*.jpg"))
        else:
            ax_files = []

        if use_planes['coronal']:
            cor_dir = os.path.join(dirs['coronal'], pid)
            cor_files = sorted(glob.glob(cor_dir + "/*.jpg"))
        else:
            cor_files = []

        if use_planes['sagittal']:
            sag_dir = os.path.join(dirs['sagittal'], pid)
            sag_files = sorted(glob.glob(sag_dir + "/*.jpg"))
        else:
            sag_files = []

        max_len = max(len(ax_files), len(cor_files), len(sag_files))

        for i in range(max_len):
            if use_planes['axial'] and i < len(ax_files):
                X['axial'].append(preprocess(ax_files[i]))
            if use_planes['coronal'] and i < len(cor_files):
                X['coronal'].append(preprocess(cor_files[i]))
            if use_planes['sagittal'] and i < len(sag_files):
                X['sagittal'].append(preprocess(sag_files[i]))

            y.append(row['ga_days'])

    for k in X:
        if len(X[k]) > 0:
            X[k] = np.array(X[k])
        else:
            X[k] = None

    return X, np.array(y)

def train(excel_path, dirs, use_planes):
    X, y = load_dataset(excel_path, dirs, use_planes)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mae_list, r2_list = [], []

    reference_plane = 'axial' if use_planes['axial'] else 'coronal' if use_planes['coronal'] else 'sagittal'

    for fold, (tr, te) in enumerate(kf.split(X[reference_plane], y // 7), 1):
        model, base = build_multiplanar_vgg16(
            {
                'axial': (224, 224, 3),
                'coronal': (224, 224, 3),
                'sagittal': (224, 224, 3)
            },
            use_planes
        )

        base.trainable = True
        for layer in base.layers[:-100]:
            layer.trainable = False

        model.compile(optimizer=Adam(1e-4), loss=Huber(), metrics=['mae'])

        train_inputs = []
        test_inputs = []

        if use_planes['axial']:
            train_inputs.append(X['axial'][tr])
            test_inputs.append(X['axial'][te])

        if use_planes['coronal']:
            train_inputs.append(X['coronal'][tr])
            test_inputs.append(X['coronal'][te])

        if use_planes['sagittal']:
            train_inputs.append(X['sagittal'][tr])
            test_inputs.append(X['sagittal'][te])

        if len(train_inputs) == 1:
            train_inputs = train_inputs[0]
            test_inputs = test_inputs[0]

        model.fit(
            train_inputs, y[tr],
            validation_data=(test_inputs, y[te]),
            epochs=50,
            batch_size=8,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=1
        )

        preds = model.predict(test_inputs).flatten()
        mae_list.append(mean_absolute_error(y[te], preds))
        r2_list.append(r2_score(y[te], preds))

    return mae_list, r2_list
