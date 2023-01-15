import cv2
from cv2 import VideoCapture
from cv2 import normalize
import numpy as np
from keras.models import load_model

import numpy as np
import PIL
import cv2
from PIL import Image, ImageTk
import pandas as pd
from tkinter import *
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
model=load_model(r'C:\Users\elifk\Desktop\Duygu_Tanıma\model_file.h5')



faceDetect=cv2.CascadeClassifier(r'C:\Users\elifk\Desktop\Facial Emotion Recognition\haarcascade_frontalface_default.xml')
duygu_dict={0:'Kizgin',1:'Igrenmis', 2:'Korkmus', 3:'Mutlu',4:'Notr',5:'Uzgun',6:'Saskın'} #dict python da bir veri tipi



master = Tk() #tkinterden bir nesne oluşturduk


arayuzu=Canvas(master, height=500,width=750,bg='light blue',)
arayuzu.pack() #ekrana yerleştirmek için bunu kullandım/Ekrana yerleştirmek için 2 fonk kullandık pack ve place
master.title("SINIFTAKİ ÖĞRENCİLER İÇİN DUYGU TESPİT SİSTEMİ") #pencerenin üstünde oluşan başlık 


image1 = PIL.Image.open("arayuz.png").convert("RGB") #image1 değişkeni oluşturup buna arayuz.png fotorğafını attım

frame_title=Frame(master,bg='light blue',) #oluşturulan frame in nerede olduğunu belirttim
frame_title.place(relx=0,rely=0,relwidth=1,relheight=0.2) #başlığın nerede olacağını belirttik
#1 yaptığımda ½100 gibi


ust_bilgilendirme_metni="\n\n\nSINIFTAKİ ÖĞRENCİLER İÇİN DUYGU TESPİT SİSTEMİ"
ust_bilgilendirme_etiketi = Label(frame_title,bg='light blue',text=ust_bilgilendirme_metni)
ust_bilgilendirme_etiketi.pack()


frame_goruntu=Frame(master,bg='light blue',)
frame_goruntu.place(relx=0,rely=0.2,relwidth=1,relheight=0.6)

arayuzfotografi = ImageTk.PhotoImage(image1) #fotoğrafı imagetk ile okudu
label_kameragoruntusu= Label(frame_goruntu,image=arayuzfotografi)
label_kameragoruntusu.image = arayuzfotografi
label_kameragoruntusu.configure(image=arayuzfotografi)
label_kameragoruntusu.place(relx=0.14,rely=0,relwidth=0.72,relheight=1)
video=cv2.VideoCapture(0)
def open_camera():   
        _,frame = video.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= faceDetect.detectMultiScale(opencv_image, 1.3, 3)
        for x,y,w,h in faces: #yüz üzerine dikdörtgen yapıyor
            sub_face_img=opencv_image[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=model.predict(reshaped) #kameradan alınan görüntü üzerinden 
            label=np.argmax(result, axis=1)[0]
            print( duygu_dict[label])            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(255,0,0),-1)
            cv2.putText(frame, duygu_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        captured_image = PIL.Image.fromarray(frame)
        photo_image = ImageTk.PhotoImage(image=captured_image)	
        label_kameragoruntusu.photo_image = photo_image
        label_kameragoruntusu.configure(image=photo_image)
        label_kameragoruntusu.after(10, open_camera)
        


def close_camera(): #kamera kapatan fonksiyon
    video.release()
    master.destroy()
    
    
alt_frame=Frame(master,bg='light blue',)
alt_frame.place(relx=0,rely=0.8,relwidth=1,relheight=0.2)

goruntu_al=Button(alt_frame, text='Görüntü Al',command=open_camera)
goruntu_al.place(relx=0.22,rely=0.3,relwidth=0.2,relheight=0.3)

sonlandir=Button(alt_frame, text='Sonlandır',command=close_camera)
sonlandir.place(relx=0.58,rely=0.3,relwidth=0.2,relheight=0.3)

master.mainloop()
