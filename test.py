from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
import pickle

import os
import csv
import time
from datetime import datetime

# install espeak, pyttsx3
# from win32com.client import Dispatch # this is for windows

# def speak(str1):
#     speak = Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

video = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

with open("data/names.pkl", 'rb') as f:
    LABELS = pickle.load(f)
    
with open("data/faces_data.pkl", 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)
imgBackground = cv.imread('background.png')

COL_NAMES = ["NAME", "TIME"]

while True:
    ret, frame = video.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5) # image, scale_factor, minNeighbours
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+h]
        resized_img = cv.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        data = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        
        exist = os.path.isfile("Attendance/Attendance_" + data + ".csv")
        
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1) 
        cv.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2) 
        cv.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1) 
        
        attendance = [str(output[0]), str(timestamp)]
        
        cv.putText(frame, # img: Matric
                   str(output[0]), # text: Any
                   (x, y-15), #org: Any -> bottom left corner of the text string in the image
                   cv.FONT_HERSHEY_COMPLEX, # fontface: Any
                   1, # fonstscale
                   (255, 255, 255), # color
                   1 # thickness
                   )
        
        cv.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1) 
    
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv.imshow("Frame", imgBackground) 
    
    k = cv.waitKey(1) 
    
    if k == ord("o"):
        # speak("Attendance Taken...")
        # time.sleep(3)
        if exist:
            with open("Attendance/Attendance_" + data + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + data + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord("q"): 
        break

video.release() 
cv.destroyAllWindows() 
      