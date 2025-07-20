import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import Toplevel

# Load the trained model
model = tf.keras.models.load_model('ecg_cnn_model.h5')

# Class labels (ensure correct order)
class_names = ['Abnormal Heartbeat', 'History of MI', 'Myocardial Infarction', 'Normal']

# Function to evaluate the model and generate confusion matrices for each class
def evaluate_model(test_generator):
    # Predict on the test set
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute the confusion matrix for the overall dataset
    cm = confusion_matrix(y_true, y_pred_classes)

    # Create a new window for the confusion matrices
    cm_window = Toplevel()
    cm_window.title("Confusion Matrices for Each Class")

    # Plot confusion matrix for each class independently
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Create a grid for 4 plots

    # Loop through each class and plot confusion matrix
    for i in range(len(class_names)):
        ax = axes[i // 2, i % 2]  # Choose the correct subplot
        cm_class = np.zeros((2, 2))  # Initialize confusion matrix for binary classification

        # Get the indices for the current class
        true_class = (y_true == i)
        pred_class = (y_pred_classes == i)

        cm_class[0, 0] = np.sum(true_class & pred_class)  # True positive
        cm_class[0, 1] = np.sum(true_class & ~pred_class)  # False negative
        cm_class[1, 0] = np.sum(~true_class & pred_class)  # False positive
        cm_class[1, 1] = np.sum(~true_class & ~pred_class)  # True negative

        sns.heatmap(cm_class, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Pred: Normal', 'Pred: Abnormal'],
                    yticklabels=['True: Normal', 'True: Abnormal'], ax=ax)
        ax.set_title(f'Confusion Matrix for {class_names[i]}')

    # Adjust layout to make room for all the subplots
    plt.tight_layout()
    plt.show()

# Create ImageDataGenerator for test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'ecg_data',  # Updated to use the same data directory as training/validation
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

# Start Tkinter root window (hidden)
root = tk.Tk()
root.withdraw()

# Evaluate the model and display confusion matrix for each class
evaluate_model(test_generator)
