from cgi import test
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os 
import scipy
import matplotlib.pyplot as plt
#gerekli kütüphaneleri import ettim.

train_data_dir='data/train/' #eğitim verilerinin yolu
validation_data_dir='data/test/'#validation train(eğitim) içinden seçili, buradan küçük bir bölüm validation olarak tanımlanır.

#eğitim kümesindeki verilere çeşitli dönüşümler uyguluyoruz, bir nevi veri önişleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3, #yakınlaştırma oranı
    horizontal_flip=True, #dik
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255) #renkler 0 ila 255 arasında ifade ediliyor.
#eğitim kısmı
train_generator = train_datagen.flow_from_directory( #dönüşüm yaptık ve değişkene atadık
                     train_data_dir, #eğitim kümesine ait dizinin yolu
                     color_mode='grayscale', #gri seviyeye çevirdik
                     target_size=(48, 48), #hedef boyutumuz
                     batch_size=32, #modele aynı anda verilen görüntü 
                     class_mode='categorical', #sınıf modu kategorik çünkü 7 sınıf mevcut.
                     shuffle=True) #karışım
#Veriler, yüzlerin 48x48 piksel gri tonlamalı görüntülerinden oluşur-->Fer2013

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir, #doğrulama kümesine ait dizinin yolu
                            color_mode='grayscale',
                            target_size=(48, 48), #Verisetindeki görüntüden fer2013
                            batch_size=32,#küçük olması iyi bir şey.modele aynı anda verilen görüntü
                            class_mode='categorical',#sınıf modumuzu kategorik koydum çünkü bu bir ikili sınıflandırma değil,7 sınıf olduğu için kategorik seçtim.
                            shuffle=True)    #rastgele seçimi sağlıyor, veri setini bölerken


class_labels=['Angry','Disgust', 'Fear', 'Happy','Natural', 'Sad','Surprise']    #sınıflar etiketlendi,                                        


img, label = train_generator.__next__() #eğitim veri seti üretilmesi, bu bizim iki veri üretmemize yardımcı oldu
#birincisi görüntü ve hangi seviyenin olduğu

#model oluşturma aşaması, ayrıca eğitimin en önemli kısmı
model = Sequential() #Sequential kerasta model olulturmanın en kolay yoludur,katman ile model olulturmamıza izin verir katmanlar ağırlıklara sahip

model.add(Conv2D(32, kernel_size=(3, 2), activation='relu', input_shape=(48,48,1))) #relu nöronları aynı anda aktive etmiyor

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary()) 

train_path ="data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs,files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root,dirs, files in os.walk(test_path):
    num_test_imgs += len(files) 
    print(num_train_imgs)
    print(num_test_imgs)
    
    epoch=30

    history=model.fit(train_generator,
                   steps_per_epoch=num_train_imgs//32,
                   epochs=epoch,
                   validation_data=validation_generator,
                   validation_steps=num_test_imgs//32)
model.save('model_file.h5')

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.xlabel("Epochs")
plt.ylabel("Acc")
