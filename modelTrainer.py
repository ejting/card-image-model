import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 


from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
import pathlib

CONST_WIDTH = 32
CONST_HEIGHT = 32

data_dir = pathlib.Path("./images/compressed")
BATCH_SIZE = 2



image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images found: {image_count}")
if image_count == 0:
    print("Warning: No images found. Check your dataset path and format.")
    all_files = list(data_dir.glob('*/*'))
    print(f"Found files (first 5): {[str(f) for f in all_files[:5]]}")

  

model = Sequential([
	layers.Input((CONST_HEIGHT, CONST_WIDTH, 1)), 
	layers.Conv2D(16, 3, padding='same'),
	layers.Conv2D(32, 3, padding='same'), 
	layers.MaxPooling2D(), 
	layers.Flatten(), 
	layers.Dense(10),
]) 

ds_train = tf.keras.preprocessing.image_dataset_from_directory( 
	data_dir,
    labels="inferred",
    label_mode = "int",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=(CONST_HEIGHT, CONST_WIDTH), # Reshape if not in this size
	shuffle=True,
	seed=123,
    validation_split=0.1, # 10% of images going to validation
    subset="training",
	) 

class_names = ds_train.class_names 
num_classes = len(class_names) 
print(class_names)
# plt.figure(figsize=(10, 10)) 
# for images, labels in ds_train.take(1): 
# 	for i in range(10): 
# 		ax = plt.subplot(5, 5, i + 1) 
# 		plt.imshow(images[i].numpy().astype("uint8")) 
# 		plt.title(class_names[labels[i]]) 
# 		plt.axis("off")


ds_validation = tf.keras.preprocessing.image_dataset_from_directory( 
	data_dir,
    labels="inferred",
    label_mode = "int",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=(CONST_HEIGHT, CONST_WIDTH), # Reshape if not in this size
	shuffle=True,
	seed=123,
    validation_split=0.1, # 10% of images going to validation
    subset="validation",
	) 

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

ds_train = ds_train.map(augment)

model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
	loss=[
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ], 
	metrics=["accuracy"],
 ) 


epochs = 10

history = model.fit( 
    ds_train,
    epochs=epochs,
    verbose=2, 
)

model.summary()

acc = history.history['accuracy'] 

loss = history.history['loss'] 
epochs_range = range(epochs) 
plt.figure(figsize=(8, 8)) 
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 

plt.legend(loc='lower right') 
plt.title('Training and Validation Accuracy') 
plt.subplot(1, 2, 2) 
plt.plot(epochs_range, loss, label='Training Loss') 

plt.legend(loc='upper right') 
plt.title('Training and Validation Loss') 
plt.show() 

model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
    print("Wrote file")

model.save("./model.keras")

print("done")