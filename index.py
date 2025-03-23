# This code is for detecting the distance of the person from the camera
# It uses the face detection and age and gender prediction to calculate the distance
# It uses the haarcascade_frontalface_default.xml for face detection
# It uses the deploy_age.prototxt and age_net.caffemodel for age prediction
# It uses the deploy_gender.prototxt and gender_net.caffemodel for gender prediction





import cv2
import numpy as np

# Constants
# If you want to change the distance, you can change the ref_image and the Known_distance
Known_distance = 60  # distance from camera to object(face) measured in centimeter
Known_width = 16.3   # width of face in the real world (centimeter)

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load detectors and models
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

# Labels
age_labels = ['0-2', '4-6', '8-12', '15-20', '21-25', '25-32',"32-38", '38-43', '48-53', '60+']
gender_labels = ['Male', 'Female']

def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_width = w
        
        # Prepare face for age and gender prediction
        face_img = image[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4, 87.7, 114.9), swapRB=False)
        
        # Predict gender
        gender_net.setInput(face_blob)
        gender = gender_labels[gender_net.forward().argmax()]
        
        # Predict age
        age_net.setInput(face_blob)
        age = age_labels[age_net.forward().argmax()]
        
        # Draw rectangle and labels
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        label = f"{gender}, {age}"
        cv2.putText(image, label, (x, y - 10), fonts, 0.8, GREEN, 2)
        
    return face_width

# Read reference image and calculate focal length
ref_image = cv2.imread("Ref_img.png")
ref_image_face_width = face_data(ref_image)
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    mirror_frame = cv2.flip(frame, 1)
    
    face_width_in_frame = face_data(mirror_frame)
    
    if face_width_in_frame != 0:
        # Calculate and display distance
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        
        # Draw distance text
        cv2.line(mirror_frame, (30, 30), (230, 30), RED, 32)
        cv2.line(mirror_frame, (30, 30), (230, 30), BLACK, 28)
        cv2.putText(mirror_frame, f"Distance: {round(Distance,2)} CM", (30, 35), 
                    fonts, 0.6, GREEN, 2)
    
    cv2.imshow("Face Detection", mirror_frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()