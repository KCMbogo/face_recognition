##### STEPS #######
# create a camera to capture
# create an infinite loop
# read what the camera captures
# show the captured image or plot the image with title and the frame
# wait for keyboard input to quit the window
# check if the key is correct then break the loop
# release the resources
# destroy all windows
# load a Cascade classifier with a model from data
# turn the detected image to gray
# detect all the faces in the frame using detectMultiScale
# iterate on all faces and draw a rectangle on each frame
# crop the face
# resize the face
# store each resized face to an array


import cv2 as cv
import numpy as np
import pickle
import os

video = cv.VideoCapture(0)

# CascadeClassifier is used to detect objects faces, eyes, cars, etc in img or video streams
# they work well with grayscale images
face_detect = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

faces_data = []
i = 0

name = input("Enter your name: ")

# while iterates in each frame
while True:
    ret, frame = video.read() # ret-> bool (if webcam opened or not) frame-> snapshot of the window
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # returns a list of faces in a rectangle form where each rect is a tuple (x, y, w, h)
    faces = face_detect.detectMultiScale(gray, 1.3, 5) # image, scale_factor, minNeighbours
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+h]
        resized_img = cv.resize(crop_img, (50, 50))
        
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
            
        i+=1
        cv.putText(frame, str(len(faces_data)), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1) # draws a rectangle to each frame
    
    cv.imshow("Frame", frame) # this shows the captured image on frame
    
    k = cv.waitKey(1) # wait for a keypress to quit in milliseconds and returns the ascii code of the pressed
    
    if k == ord("q"): # ord() returns the ascii code for the letter
        break

video.release() # release the software and hardware resources
cv.destroyAllWindows() # close all opened windows
      
      
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)      
      
if "names.pkl" not in os.listdir('data/'):
    names = [name] * len(faces_data) 
    with open("data/names.pkl", 'wb') as f:
        pickle.dump(names, f) 
else:    
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
    names = names + [name] * len(faces_data)
    with open("data/names.pkl", 'wb') as f:
        pickle.dump(names, f)
        
if "face_data.pkl" not in os.listdir('data/'):
    with open("data/faces_data.pkl", 'wb') as f:
        pickle.dump(faces_data, f) 
else:    
    with open("data/faces_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data)
    with open("data/faces_data.pkl", 'wb') as f:
        pickle.dump(faces_data, f)
    