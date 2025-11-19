import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def extract_attention(model, img):
    layers = [l for l in model.layers if 'conv2d' in l.name]
    target = layers[-1]
    att_model = Model(inputs=model.inputs, outputs=target.output)
    fmap = att_model.predict(img)[0]
    heat = np.mean(fmap, axis=-1)
    heat = cv2.resize(heat, (224, 224))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat

def overlay(img_path, heat):
    orig = img_to_array(load_img(img_path, target_size=(224, 224))).astype('uint8')
    heatmap = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    out = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
    return out

def save_heatmap(model, img_path, out_path):
    img = preprocess(img_path)
    heat = extract_attention(model, img)
    result = overlay(img_path, heat)
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
