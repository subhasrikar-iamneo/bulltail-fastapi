import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model
from keras.metrics import MeanAbsoluteError
from PIL import Image
import numpy as np

# Define the custom object dictionary
custom_objects = {
    'mae': MeanAbsoluteError()
}

# Load the model with custom objects
model = load_model('my_model2.h5', custom_objects=custom_objects)

def predicage(model=model):
    img = load_img('face.png', color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img, dtype=float)
    img /= 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, H, W, C)
    
    pred=model.predict(img)
    pred_gender=(pred[0]>0.5)*1
    pred_age = [round(pred[0]) for pred in pred[1]]
    gear = ["Male","Female"]
    return(gear[pred_gender[0][0]], str(pred_age[0]))
