import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, Reshape, Multiply, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def attention_module(x):
    att = Conv2D(x.shape[-1], (1, 1), activation='sigmoid')(x)
    att = Reshape((x.shape[1], x.shape[2], x.shape[3]))(att)
    out = Multiply()([x, att])
    return out

def build_multiplanar_vgg16(input_shapes, use_planes):
    inputs = []
    features = []

    base = VGG16(weights='imagenet', include_top=False)

    if use_planes['axial']:
        inp_ax = Input(shape=input_shapes['axial'], name='axial_input')
        f_ax = base(inp_ax)
        f_ax = attention_module(f_ax)
        f_ax = Flatten()(f_ax)
        inputs.append(inp_ax)
        features.append(f_ax)

    if use_planes['coronal']:
        inp_cor = Input(shape=input_shapes['coronal'], name='coronal_input')
        f_cor = base(inp_cor)
        f_cor = attention_module(f_cor)
        f_cor = Flatten()(f_cor)
        inputs.append(inp_cor)
        features.append(f_cor)

    if use_planes['sagittal']:
        inp_sag = Input(shape=input_shapes['sagittal'], name='sagittal_input')
        f_sag = base(inp_sag)
        f_sag = attention_module(f_sag)
        f_sag = Flatten()(f_sag)
        inputs.append(inp_sag)
        features.append(f_sag)

    merged = features[0] if len(features) == 1 else concatenate(features)

    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(merged)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=out)

    return model, base
