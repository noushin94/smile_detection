import os
from mtcnn import MTCNN
import cv2
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras import models, layers
import matplotlib.pyplot as plt

detector = MTCNN()

def detect_faces(img):
   
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y , w, h = out["box"]
    return rgb_img[y:y+h , x:x+w] # box of the face
all_faces = []
all_labels = []

for i,item in enumerate(glob.glob("/Users/noushinahmadvand/Documents/smile detection/smile_dataset/*/*")):
    img = cv2.imread(item)
    face = detect_faces(img)
    face = cv2.resize(face, (32,32))
    face = face/255.0
    all_faces.append(img)

     #print(address.split("/")[-2])
    labels = item.split("/")[-2]

    all_labels.append(labels)

    if i % 100 == 0 :
          print(f"[INFO] {i}/4000 processed")

   
 

features_vectors = np.array(all_faces)          

lb = LabelBinarizer()

all_labels =  lb.fit_transform(all_labels)






#print(label_encoded)

# spliting data into train and test

X_train , X_test , Y_train , Y_test = train_test_split(features_vectors, all_labels, test_size= 0.2)


# defing the CNN network

net = models.Sequential([ 
     
                         layers.Conv2D(32, (3,3), activation = "relu", input_shape = (32,32,3)),
                         layers.Conv2D(32, (3,3), activation = "relu"),
                         layers.MaxPooling2D((2,2)),

                         layers.Conv2D(64, (3,3), activation = "relu"),
                         layers.Conv2D(64, (3,3), activation = "relu"),
                         layers.MaxPooling2D((2,2)),

                         layers.Flatten(),
                         layers.Dense(100, activation='relu'),
                         layers.Dense(2, activation= 'softmax')
     
                         ])


print(net.summary())

net.compile(optimizer= "SGD",
          loss= "categorical_crossentropy",
        metrics = ["accuracy"])




h = net.fit(X_train,Y_train, batch_size=32, validation_data= (X_test, Y_test), epochs = 10)


loss , acc = net.evaluate(X_test, Y_test)


 # it by itself evaluate main y and compare it with y predict





plt.plot(h.history["accuracy"], label = "train accuracy" )
plt.plot(h.history["val_accuracy"], label = "test accuracy")
plt.plot(h.history["loss"], label = "train_accuracy" )
plt.plot(h.history["val_loss"], label = "test accuracy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("tumor detection")
plt.show()