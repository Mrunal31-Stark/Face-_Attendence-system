import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
faces_d = cv2.CascadeClassifier("/haarcascade_frontalface_default.xml")
fac_data = []
i = 0

name = input("Enter Name: ")
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_d.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize = cv2.resize(crop_img, (80, 80))
        if len(fac_data) <= 100 and i % 10 == 0:
            fac_data.append(resize)
        i = i + 1
        cv2.putText(frame, str(len(fac_data)), (80, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (80, 80, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 255), 1)
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(fac_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

fac_data = np.asarray(fac_data)
fac_data = fac_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('C:/Users/Acer/Desktop/FACE12/'):
    names = [name] * 100
    with open('C:/Users/Acer/Desktop/FACE12/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('C:/Users/Acer/Desktop/FACE12/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('C:/Users/Acer/Desktop/FACE12/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('C:/Users/Acer/Desktop/FACE12/'):
    with open('C:/Users/Acer/Desktop/FACE12/faces_data.pkl', 'wb') as f:
        pickle.dump(fac_data, f)
else:
    with open('C:/Users/Acer/Desktop/FACE12/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
        faces = np.append(faces, fac_data, axis=0)
    with open('C:/Users/Acer/Desktop/FACE12/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
