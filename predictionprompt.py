import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

# Load trained model
model = tf.keras.models.load_model('ecg_cnn_model.h5')

# Class labels (ensure order matches training)
class_names = ['Abnormal Heartbeat', 'History of MI', 'Myocardial Infarction', 'Normal']

# Function to detect ECG condition
def detect_ecg(image_path):
    img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Show ECG image
    plt.imshow(img_array[0].squeeze(), cmap='gray')
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)

    return predicted_label, confidence

# Ask for patient details
print("\n👤 Patient Information")
name = input("Enter patient name: ")
age = input("Enter age: ")
gender = input("Enter gender (M/F/Other): ")
symptoms = input("Describe any symptoms (optional): ")

# Ask for ECG image
print("\n📂 ECG Image")
image_path = input("Enter full path to ECG image: ").strip('"')

if os.path.exists(image_path):
    # Detect ECG condition
    condition, confidence = detect_ecg(image_path)

    # Suggestion message
    if condition == 'Normal':
        suggestion = "🟢 No abnormalities detected. ECG appears normal."
    elif condition == 'Abnormal Heartbeat':
        suggestion = "🟠 Irregular heartbeat detected. Consider visiting a cardiologist."
    elif condition == 'History of MI':
        suggestion = "🟡 Signs of previous myocardial infarction. Periodic follow-up recommended."
    elif condition == 'Myocardial Infarction':
        suggestion = "🔴 Possible current myocardial infarction. Seek emergency care immediately!"

    # Print diagnostic report
    print("\n📋 ================= ECG DIAGNOSTIC REPORT =================")
    print(f"📅 Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Name       : {name}")
    print(f"🎂 Age        : {age}")
    print(f"⚧ Gender     : {gender}")
    if symptoms:
        print(f"🩺 Symptoms   : {symptoms}")
    print(f"\n🧠 Prediction : {condition}")
    print(f"📊 Confidence : {confidence:.2f}%")
    print(f"💬 Suggestion : {suggestion}")
    print("===========================================================\n")

else:
    print("❌ Error: File path is incorrect. Please check and try again.")
C