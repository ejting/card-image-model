import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf
from PIL import Image
from pillow_heif import register_heif_opener
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import pathlib


def get_image_dimensions(image_path):
    """
    Get the width and height of an image from its path.

    Args:
      image_path: The path to the image file.

    Returns:
      A tuple containing the width and height of the image.
      Returns None if the image cannot be opened.
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width, height
    except FileNotFoundError:
        print(f"Error: Image not found at path: {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def preprocess_image(image_path, target_size):
  img = image.load_img(image_path, target_size=target_size)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0) # Or np.array([img_array])
  img_array /= 255. # Normalization
  return img_array

def predict_image(model, image_path, target_size):
  processed_image = preprocess_image(image_path, target_size)
  prediction = model.predict(processed_image)
  return prediction

def interpret_prediction(prediction, class_names=None):
  if class_names:
    
    predicted_class_index = np.argmax(prediction)
    print(prediction)
    predicted_class = class_names[predicted_class_index]
    return predicted_class
  else:
    return prediction

CONST_WIDTH = 32
CONST_HEIGHT = 32


register_heif_opener()


model = tf.keras.models.load_model("./model.keras")
img_path = "./images/raw/7s/7s_3.HEIC"

target_size = (32, 32) # Example dimensions, adjust to your model's expected input
class_names = ['7s', '8c', '8s', 'ah']  # Replace with your actual class names

prediction = predict_image(model, img_path, target_size)
result = interpret_prediction(prediction, class_names)


if class_names:
  print(f'Predicted class: {result}')
else:
  print(f'Prediction: {result}')
# if(dimensions):
#     width, height = dimensions
    
#     print(f"width: {width}, height: {height}")

#     image = tf.keras.preprocessing.image.load_img(img_path, target_size=(height, width))
#     image_array = tf.keras.preprocessing.image.img_to_array(image)
#     preprocessed_image = np.expand_dims(image_array, axis=0)
#     preprocessed_image = preprocessed_image / 255.0 # Normalize pixel values

#     # Make predictions
#     prediction = model.predict(preprocessed_image)

#     # Interpret the results for a classification model
#     predicted_class_index = np.argmax(prediction)
    
#     predicted_class_name = class_names[predicted_class_index]

#     print(f"Predicted class: {predicted_class_name}")