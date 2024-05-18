from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)


video = cv2.VideoCapture(0)
faces_d = cv2.CascadeClassifier("/haarcascade_frontalface_default.xml")

with open('C:/Users/Acer/Desktop/FACE12/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('C:/Users/Acer/Desktop/FACE12/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure the number of samples in FACES matches the number of samples in LABELS
# If they don't match, adjust your data loading process accordingly.

# Check the number of samples in each dataset
print("Number of samples in FACES:", len(FACES))
print("Number of samples in LABELS:", len(LABELS))

# If they are not equal, you need to investigate why and adjust your data loading process.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)
background = cv2.imread("Design4.png")
col_names = ["Names","Time"]
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_d.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize = cv2.resize(crop_img, (80, 80)).flatten().reshape(1, -1)
        output = knn.predict(resize)
        ts =time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile("C:/Users/Acer/Desktop/FACE12/ATTENDENCE/ATTENDENCE_file"+date+".csv")
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x + w, y), (80, 80, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 255), 1)
        attendence = [str(output[0]),str(timestamp)]
    background[162:162+480,55:55+640] = frame
    cv2.imshow("frame", background)
    k = cv2.waitKey(1)
    if k==ord('o'):
        speak("Hello Sir ! , Your Attendence is Marked ..")
        time.sleep(5)
        if exist:
            with open("C:/Users/Acer/Desktop/FACE12/ATTENDENCE/ATTENDENCE_file" + date + ".csv", "a") as csvfile:
                writers = csv.writer(csvfile)
                writers= writers.writerow(attendence)
        else:
            with open("C:/Users/Acer/Desktop/FACE12/ATTENDENCE/ATTENDENCE_file"+date+".csv","a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(col_names)
                writer.writerow(attendence)
            csvfile.close()
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
