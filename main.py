# Import necessary libraries
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to build the CNN model
def build_cnn_model(input_shape=(64, 64, 3), num_classes=10):
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening the layers
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Image data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Directory paths for your dataset (update these paths)
train_dir = 'path/to/train'
validation_dir = 'path/to/validation'

# Loading the dataset
train_set = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
validation_set = validation_datagen.flow_from_directory(validation_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')

# Build and compile the model
model = build_cnn_model(input_shape=(64, 64, 3), num_classes=train_set.num_classes)

# Train the model
history = model.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=20,
    validation_data=validation_set,
    validation_steps=len(validation_set)
)

# Plotting accuracy and loss over epochs
def plot_history(history):
    # Accuracy plot
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Loss plot
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot training history
plot_history(history)

# Function to predict the species in a new image
def predict_species(image_path, model, classes):
    # Load the image and resize it to match the model's input shape
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0  # Rescale as in training

    # Predict
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    class_name = classes[class_index]
    
    print(f"Predicted Species: {class_name}")

# Usage example (update image path)
classes = list(train_set.class_indices.keys())  # Get class names from training set
predict_species('path/to/new_image.jpg', model, classes)

# Save the model after training
model.save('wildlife_cnn_model.h5')
