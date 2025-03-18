import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pytesseract
from PIL import Image
from gtts import gTTS

# Set the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to convert text to speech and save as output.mp3
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    output_file = "output.mp3"
    tts.save(output_file)
    return output_file

# Load class names from a file
def load_class_names():
    with open("coco.txt", "r") as f:  # Adjust the path to your class names file
        class_names = f.read().strip().split("\n")
    return class_names

# Count the number of occurrences of each object class
def count_objects(predicted_classes):
    class_counts = {}
    for cls in predicted_classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    return class_counts

# Convert the count of objects into a text format
def convert_counts_to_text(class_counts):
    text = "Predicted objects: "
    for cls, count in class_counts.items():
        text += f"{count} {cls}, "
    return text[:-2]  # Remove the last comma and space

# Process the uploaded image with YOLO
def process_image(image):
    # Convert PIL Image to a NumPy array
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Change from RGB to BGR
    model = YOLO('yolov8n.pt')  # Path to the YOLO model (adjust if needed)
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    class_names = load_class_names()  # Load the class names
    predicted_classes = []  # Initialize a list to store predicted class names

    # Loop through the results and collect the predicted class names
    for index, row in px.iterrows():
        _, _, _, _, _, d = row
        class_name = class_names[int(d)] if int(d) < len(class_names) else "Unknown"
        predicted_classes.append(class_name)  # Append class names to the list

    # Call the new function to count the occurrences of each object class
    class_counts = count_objects(predicted_classes)

    return class_counts  # Return the object counts (dictionary) instead of the raw class names

# Extract text from image using Tesseract
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image).strip()
    return text if text else None  # Return None if no text found

# Streamlit App
st.title("ENCHANCING THE LIVES OF VISUALLY IMPARIED PEOPLE")

# Login Page
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Admin" and password == "123":
            st.session_state.logged_in = True
            st.success("Logged in as Admin")
        else:
            st.error("Invalid username or password")

# Main App Functionality
if st.session_state.logged_in:
    st.subheader("Capture an Image")
    camera_input = st.camera_input("Capture an image")

    if camera_input is not None:
        image = Image.open(camera_input)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Process Image
        if st.button("Process Image"):
            st.write("Processing...")
            class_counts = process_image(image)  # Get the object counts from the image
            extracted_text = extract_text_from_image(image)

            # Check results and display appropriate messages
            if class_counts and extracted_text:
                result_text = convert_counts_to_text(class_counts) + f". Recognized text: {extracted_text}"
                st.write(result_text)
                audio_output = text_to_speech(result_text)  # Convert result to speech
            elif class_counts:
                result_text = convert_counts_to_text(class_counts) + ". No text found."
                st.write(result_text)
                audio_output = text_to_speech(result_text)
            elif extracted_text:
                result_text = f"No objects found. Recognized text: {extracted_text}"
                st.write(result_text)
                audio_output = text_to_speech(result_text)
            else:
                result_text = "No objects or text found."
                st.write(result_text)
                audio_output = text_to_speech(result_text)

            # Play the audio output
            st.audio(audio_output)  # Stream audio output
