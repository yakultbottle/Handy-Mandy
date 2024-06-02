import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings

# Suppress warnings in stdout
warnings.filterwarnings('ignore', category=Warning)

# Check TensorFlow version
print(f'TensorFlow Version: {tf.__version__}')

# Get the absolute path to the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the dataset directory
dataset_dir = os.path.join(base_dir, 'leapGestRecog')

# Print the dataset directory path to ensure it's correct
print(f"Dataset Directory: {dataset_dir}")

# Create an ImageDataGenerator for loading and augmenting images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of the data will be used for validation
)

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Calculate steps per epoch
total_training_samples = len(train_generator.filenames)
total_validation_samples = len(validation_generator.filenames)
batch_size = 32
steps_per_epoch = total_training_samples // batch_size
validation_steps = total_validation_samples // batch_size

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Check the shape of a batch from the training generator
for images, labels in train_generator:
    print(f"Training Batch Shape - Images: {images.shape}, Labels: {labels.shape}")
    break

# Convert generators to tf.data.Dataset
def generator_to_dataset(generator):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(generator.class_indices)), dtype=tf.float32)
        )
    )
    return dataset

train_dataset = generator_to_dataset(train_generator)
validation_dataset = generator_to_dataset(validation_generator)

# Optimize data pipeline
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
    Dense(len(train_generator.class_indices), activation='softmax', dtype=tf.float32)  # Ensure output layer uses float32
])

# Print model summary
model.summary()

# Compile the model with an initial learning rate
initial_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks for early stopping, model checkpointing, and learning rate reduction
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=False),  # Ensure training continues
    ModelCheckpoint('gesture_recognition_model.keras', save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6, verbose=1)
]

# Train the model with callbacks
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    epochs=50,  # Increase the number of epochs
    callbacks=callbacks,
    verbose=2  # Increase verbosity to see more detailed output
)

# Evaluate the model on the validation set after training
print("Evaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Print final training accuracy for comparison
print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")

