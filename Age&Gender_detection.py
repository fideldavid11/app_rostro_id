import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('style.xml')
video_capture.set(3, 480)
video_capture.set(4, 640)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Hombre', 'Mujer']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def video_detector(age_net, gender_net, video_source=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    root = tk.Tk()
    root.title("App reconocimiento de rostro")

    label = ttk.Label(root, text="App reconocimiento de rostro")
    label.pack(padx=10, pady=10)

    video_panel = tk.Label(root)
    video_panel.pack(padx=10, pady=10)

    def update_frame():
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 10, cv2.CASCADE_SCALE_IMAGE, (30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (244, 244), MODEL_MEAN_VALUES, swapRB=True)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            overlay_text = f"{gender}, {age}"
            cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.configure(image=imgtk)
        video_panel.after(10, update_frame)

    update_frame()

    root.mainloop()

def main():
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)

if __name__ == "__main__":
    main()

