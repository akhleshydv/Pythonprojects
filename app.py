# this is the user interface version of the age, gender and distance finder
# it is a streamlit app that uses the webrtc_streamer to stream the video from the webcam
#you can use index.py for the proper code logic and implementation
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time

# Load models and constants
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

age_labels = ['0-2', '4-6', '8-12', '15-20', '21-25', '25-32', '32-38', '38-43', '48-53', '60+']
gender_labels = ['Male', 'Female']

Known_distance = 60  # cm
Known_width = 16.3   # cm
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
fonts = cv2.FONT_HERSHEY_COMPLEX

# Helper functions
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame

def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        face_width = w
        face_img = image[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4, 87.7, 114.9), swapRB=False)
        gender_net.setInput(face_blob)
        gender = gender_labels[gender_net.forward().argmax()]
        age_net.setInput(face_blob)
        age = age_labels[age_net.forward().argmax()]
        results.append({
            'rect': (x, y, w, h),
            'gender': gender,
            'age': age,
            'face_width': w
        })
    return results

# Calculate focal length using reference image
ref_image = cv2.imread("Ref_img.png")
ref_faces = face_data(ref_image)
ref_image_face_width = ref_faces[0]['face_width'] if ref_faces else 1  # Avoid division by zero
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_data(img)
        result_text = ""
        for face in faces:
            x, y, w, h = face['rect']
            gender = face['gender']
            age = face['age']
            face_width = face['face_width']
            distance = None
            if face_width != 0:
                distance = Distance_finder(Focal_length_found, Known_width, face_width)
                cv2.line(img, (30, 30), (230, 30), RED, 32)
                cv2.line(img, (30, 30), (230, 30), BLACK, 28)
                cv2.putText(img, f"Distance: {round(distance,2)} CM", (30, 35), fonts, 0.6, GREEN, 2)
            label = f"{gender}, {age}"
            cv2.rectangle(img, (x, y), (x+w, y+h), GREEN, 2)
            cv2.putText(img, label, (x, y - 10), fonts, 0.8, GREEN, 2)
            # Save result for UI
            result_text = f"Gender: {gender}, Age: {age}, Distance: {round(distance,2) if distance else 'N/A'} CM"
        self.latest_result = result_text
        return img

def on_webrtc_ended():
    # This function is called when the stream stops
    # You can add any cleanup code here if needed
    st.info("Webcam stream ended. Camera should be released.")

st.title("Age, Gender & Distance Finder")
st.write("Open your webcam to detect age, gender, and distance from the camera.")

# Start the streamer and get the context
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer
)

result_placeholder = st.empty()

# Poll for results while the streamer is running
if webrtc_ctx.video_processor:
    while webrtc_ctx.state.playing:
        result = webrtc_ctx.video_processor.latest_result
        if result:
            result_placeholder.markdown(f"**Latest Detection:** {result}")
        else:
            result_placeholder.markdown("No face detected.")
        # Add a small delay to avoid excessive CPU usage
        time.sleep(0.1) 
