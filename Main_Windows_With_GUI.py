from tkinter import *
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
from PIL import Image;
import webbrowser




# creating instance of TK
root = Tk()

root.configure(background="white")


# root.geometry("300x300")

def function1():
    face_classifier = cv2.CascadeClassifier(
        'C:\\Users\\krabh\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            croped_face = img[y:y + w, x:x + w]
        return croped_face

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, fram = cap.read()
        if face_extractor(fram) is not None:
            count += 1
            face = cv2.resize(face_extractor(fram), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = 'E:\\SOFTWARE\\INT248Project\\venv\\Face_data_folder\\Faces\\user' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face Not Found\n")
            pass
        if cv2.waitKey(1) == 13 or count == 50:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\n\t\t Face Data Collected Successfuly Done.\n\n")


def function2():
    data_path = 'E:\\SOFTWARE\\INT248Project\\venv\\Face_data_folder\\Faces\\'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    Trainig_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Trainig_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Trainig_Data), np.asarray(Labels))
    print("\n\n\t\t Data Successfuly Trained .\n\t\t")

    # os.system("py training_dataset.py")


def function3():
    data_path = 'E:\\SOFTWARE\\INT248Project\\venv\\Face_data_folder\\Faces\\'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    Trainig_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Trainig_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Trainig_Data), np.asarray(Labels))
    # print("\n\n\t\t Data Successfuly Trained .\n\t\t")

    face_classifier = cv2.CascadeClassifier(
        'C:\\Users\\krabh\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            if result[1] < 500:
                confidance = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidance) + '%  Accuracy'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 120, 255), 2)

            if confidance > 75:
                cv2.putText(image, "Face Detected", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Wrong Face Data", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)





        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    # os.system("py recognizer.py")


def function4():
    root.destroy()

def function5():
    webbrowser.open_new(r'file://E:\SOFTWARE\INT248Project\venv\Reports.docx')



    




# setting title for the window
root.title("<<<   FACE DETECTION SYSTEM   >>>>>")

# creating a text label
Label(root, text="FACE RECOGNITION  SYSTEM", font=("times new roman", 20), fg="white", bg="maroon",
      height=2).grid(row=0, rowspan=2, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)

# creating first button
Button(root, text="Create Dataset", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function1).grid(
    row=3, columnspan=2, sticky=W + E + N + S, padx=5, pady=5)

# creating second button
Button(root, text="Train Dataset", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function2).grid(
    row=4, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)

# creating third button
Button(root, text="Recognize Faces", font=('times new roman', 20), bg="#0D47A1", fg="white",
       command=function3).grid(row=5, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)


Button(root, text="Project_Report", font=('times new roman', 20), bg="maroon", fg="white",command=function5).grid(row=6,
                                                                                                         columnspan=2,
                                                                                                         sticky=N + E + W + S,
                                                                                                         padx=5, pady=5)


Button(root, text="Exit", font=('times new roman', 20), bg="maroon", fg="white", command=function4).grid(row=9,
                                                                                                         columnspan=2,
                                                                                                         sticky=N + E + W + S,
                                                                                                         padx=5, pady=5)



root.mainloop()
