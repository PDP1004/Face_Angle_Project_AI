import cv2
import PIL
import dlib
import numpy as np
import tkinter as tk
import pickle as pkl
from tkinter import *
import matplotlib as plt
from logging import root
from select import select
from cProfile import label
from operator import truediv
from PIL import ImageTk, Image
from tkinter.filedialog import Open
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []

    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))     
    return np.array(features).reshape(1, -1)

x, y = pkl.load(open('data/samples.pkl', 'rb'))
roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

std = StandardScaler()
std.fit(x)
model = load_model('models/faceangle.h5')

# Hàm tiên đoán giá trị góc xoay khuôn mặt
def pred(imgin):
    imgpre = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
    face_points = detect_face_points(imgpre)
    try:
        features = compute_features(face_points)
        features = std.transform(features)
        
        y_pred = model.predict(features)
        roll_pred, pitch_pred, yaw_pred = y_pred[0]

        cv2.putText(imgin, 'Roll:  {:.2f}'.format(roll_pred),
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.putText(imgin, 'Pitch: {:.2f}'.format(pitch_pred),
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)                   
        cv2.putText(imgin, 'Yaw:   {:.2f}'.format(yaw_pred),
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)             
        return imgin
    except:
        pass

# Tạo giao diện 
class Application(Frame):  
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(fill=BOTH, expand=True, padx=15, pady=15)
        self.create_widgets()

    def create_widgets(self):
        self.columnconfigure(0, pad=20)      
        self.rowconfigure(0, pad=20)

        self.BT_Camera = Button(self, text="Test With Camera", fg='blue', height=2, width=15, command=self.onCamera)
        self.BT_Camera.place(x= 50, y=20)       

        self.BT_Image = Button(self, text="Test With Image", fg='blue',height=2, width=15, command=self.onImage)
        self.BT_Image.place(x= 50, y=120)

        self.BT_Video = Button(self, text="Test With Video", fg='blue',height=2, width=15, command=self.onVideo)
        self.BT_Video.place(x= 50, y=220)  

        self.BT_CamSmartphone = Button(self, text="Test With Camera \n Smartphone", fg='blue',height=2, width=15, command=self.onCamSmartphone)
        self.BT_CamSmartphone.place(x= 50, y=320) 

        img = Image.open("2022-06-07_153919.png")
        img = img.resize((350,233))
        imgshow = ImageTk.PhotoImage(img)        
        self.photo = Label(self, image = imgshow)
        self.photo.image_names = imgshow
        self.photo.place(x= 200, y=80) 
        
        self.IPCam = Entry(self, width=35,font=('calibre',13))
        self.IPCam.place(x= 50, y=390)

        self.lb1 = Label(self, text='IP Camera:') 
        self.lb1.place(x= 50, y=365)
        self.lb1.pack_forget()

        self.BT_OK = Button(self, text="OK", fg='blue',height=1, width=3, command=self.onOK)
        self.BT_OK.place(x= 380, y=388) 
        
        self.frame = Frame(self,height=75, width=400)
        self.frame.place(x= 50, y=360)

# Hàm xử lý khi nhấn nút "Test With Camera"
    def onCamera(self):
        cap = cv2.VideoCapture(0)
        while True:
            [success, frame] = cap.read()
            frame = cv2.flip(frame,1) 
            imgout = pred(frame)
            cv2.imshow('Video Out', imgout)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

# Hàm xử lý khi nhấn nút "Test With Image"  
    def onImage(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()

        if fl != '':
            im = cv2.imread(fl, cv2.IMREAD_COLOR)
            imgout = pred(im)
            imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
            imgout = PIL.Image.fromarray(imgout)
            imgout = imgout.resize((450,350))
            imgshow = ImageTk.PhotoImage(imgout)        
            self.photo = Label(self, image = imgshow)
            self.photo.image_names = imgshow
            self.photo.place(x= 200, y=20)

# Hàm xử lý khi nhấn nút "Test With Video"            
    def onVideo(self):
        global ftypes
        ftypes = [('video', '*.mp4 *.mkv *.wmv *.flv *.vob')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
        if fl != '':
            cap = cv2.VideoCapture(fl)
            while True:
                [success, frame] = cap.read()
                frame = cv2.flip(frame,1) 
                imgout = pred(frame)
                cv2.namedWindow('Video Out', cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow('Video Out', 600, 450)
                cv2.imshow('Video Out', imgout)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
    
# Hàm xử lý khi nhấn nút "Test With Camera Smartphone"
    def onCamSmartphone(self):
        self.lb1.lift(self.frame)
        self.IPCam.lift(self.frame)
        self.BT_OK.lift(self.frame)

# Hàm xử lý khi nhấn nút "OK"
    def onOK(self):                 # Để kết nối với camera smartphone, ta dùng ứng dụng IPWebCam: https://play.google.com/store/apps/details?id=com.pas.webcam Hoặc DroidCam
        ip = self.IPCam.get()       # Định dạng địa chỉ để kết nối với Smartphone: http://192.168.1.30:8080/video
        cap = cv2.VideoCapture((ip))
        while True:
            [success, frame] = cap.read()
            frame = cv2.flip(frame,1) 
            cv2.imshow('Video Out', frame)
            imgout = pred(frame)
            cv2.imshow('Video Out', imgout)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


root = tk.Tk()
app = Application(master=root)
app.master.title("Face Angle Recognize")
app.master.geometry("600x500+300+300")
app.mainloop()   