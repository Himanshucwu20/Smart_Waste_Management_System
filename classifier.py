
# train_path = "Users\HIMANSHU\DATASET\TRAIN"
# test_path = "Users\HIMANSHU\DATASET\TEST"

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Define the path to the training and testing datasets
train_path = "C:\\Users\\HIMANSHU\\DATASET\\TRAIN"
test_path = "C:\\Users\\HIMANSHU\\DATASET\\TEST"

# Define the batch size and image dimensions
batch_size = 32
img_height = 224
img_width = 224

# Define the number of classes
num_classes = 2

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an instance of the ImageDataGenerator class for training and testing data
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1./255)

# Load the training and testing datasets using the flow_from_directory method
train_data = train_data_gen.flow_from_directory(train_path, target_size=(img_height, img_width),
                                                batch_size=batch_size, class_mode='categorical')
test_data = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width),
                                               batch_size=batch_size, class_mode='categorical')

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Save the trained model to a file
model.save('waste_classifier.h5')

# Load the saved model
model = keras.models.load_model('waste_classifier.h5')