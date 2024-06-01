import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print(tf.__version__)

# Get the absolute path to the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the dataset directory
dataset_dir = os.path.join(base_dir, 'leapGestRecog')

# Print the dataset directory path to ensure it's correct
print("Dataset Directory:", dataset_dir)

# Create an ImageDataGenerator for loading and augmenting images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create training generator (without validation split)
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Debug data loading
print("Training Generator - Batches:", len(train_generator))
print("Training Generator - Samples per batch:", train_generator.batch_size)
print("Training Generator - Total samples:", len(train_generator) * train_generator.batch_size)

# Calculate steps per epoch
total_training_samples = len(train_generator.filenames)
batch_size = 32
steps_per_epoch = total_training_samples // batch_size
print("Steps per epoch:", steps_per_epoch)

# Check the shape of a batch from the training generator
for images, labels in train_generator:
    print("Training Batch Shape - Images:", images.shape, "Labels:", labels.shape)
    break

# Build the model
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax') 
])

# Print model summary for verification
model.summary()

# Compile the model with an initial learning rate
initial_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks for early stopping, model checkpointing, and learning rate reduction
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('gesture_recognition_model.keras', save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6, verbose=1)
]

# Train the model with callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # Increase the number of epochs
    callbacks=callbacks
)
