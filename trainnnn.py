import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import zipfile
import os

# --- STEP 1: UNZIP DATASET ---
# Unzip all ECG datasets into respective folders (replace with actual paths)
def unzip_data(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# Example for unzipping (repeat this for other datasets as well):
unzip_data("D:/Downloads/Normal Person ECG Images (284x12=3408).zip", "ecg_data/normal")
unzip_data("D:/Downloads/ECG Images of Patient that have History of MI (172x12=2064).zip", "ecg_data/history_mi")
unzip_data("D:/Downloads/ECG Images of Patient that have abnormal heartbeat (233x12=2796).zip", "ecg_data/abnormal")
unzip_data("D:/Downloads/ECG Images of Myocardial Infarction Patients (240x12=2880).zip", "ecg_data/mi")

# --- STEP 2: PREPARE DATA ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Rotate images to simulate real-world ECG variability
    width_shift_range=0.1,  # Allow some horizontal shift
    height_shift_range=0.1,  # Allow some vertical shift
    zoom_range=0.1,  # Slight zoom-in for robustness
    shear_range=0.2,  # Slight shear for robustness
    horizontal_flip=True,  # Randomly flip ECG signals
    validation_split=0.2  # Use 20% of the data for validation
)

# Training and validation generators
train_generator = train_datagen.flow_from_directory(
    'ecg_data',  # Folder where all classes are stored
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',  # ECG images are single-channel (grayscale)
    subset='training'  # Use the subset for training
)

val_generator = train_datagen.flow_from_directory(
    'ecg_data',  # Folder where all classes are stored
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'  # Use the subset for validation
)

# --- STEP 3: COMPUTE CLASS WEIGHTS ---
# Compute class weights to handle class imbalance
labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weights_dict)

# --- STEP 4: BUILD CNN MODEL ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout to reduce overfitting
    tf.keras.layers.Dense(4, activation='softmax')  # 5 classes to match data folders: Normal, Abnormal, History of MI, Acute MI, mi
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- STEP 5: EARLY STOPPING CALLBACK ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- STEP 6: TRAIN THE MODEL ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # You can experiment with a higher number of epochs
    class_weight=class_weights_dict,  # Apply class weights
    callbacks=[early_stopping]  # Use early stopping to prevent overfitting
)

# --- STEP 7: SAVE THE MODEL ---
model.save('ecg_cnn_model_with_class_weights_and_early_stopping.h5')
print("Model saved as ecg_cnn_model_with_class_weights_and_early_stopping.h5")

