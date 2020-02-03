# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import classification_report,accuracy_score
#import load_folder

ds_path = "Rock-Paper-Scissors/train/"

images = []
classes = []
def load_image(folder):   
    for filename in os.listdir(folder):
        label = os.path.basename(folders)
        className = np.asarray( label )
        if label is not None:
            classes.append(className)
            #labels_hot.append(dict_labels[label])
            #np.append(labels, className , axis=0)
            #np.append(labels_hot, np.array(dict_labels[label]), axis=0)
            
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (300, 300))
        if img is not None:
            images.append(np.array(img))  
    return images, classes

#------TRAIN------------------------
for folders in glob.glob(ds_path+"/*"):
    folders = folders.replace("\\","/")
    print("Load {} ...".format(folders))
    images, classes = load_image(folders)

#images, classes = load_folder.load_ds(ds_path, resize=resize_img)
features = np.array(images, 'int16')
labels = np.array(classes)
print(features.shape, labels.shape)
list_hog_fd = []

for feature in features:
    fd = hog(feature, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(8, 8), visualize=False)
    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')
print(hog_features.shape)

X_train, X_test, y_train, y_test = train_test_split(
    hog_features,
    labels,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

clf = LinearSVC()
clf.fit(X_train, y_train)

#------EVALUATION------------------------
y_pred = clf.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

#joblib.dump(clf, "hog_svm.pkl", compress=3)

##------TESTING------------------------
## Load the classifier
#clf = joblib.load("hog_svm.pkl")
#
## Read the input image
#imgs = []
#files = []
#
#for i in range(527,540):
#    im = cv2.imread("test/IMG_0"+str(i)+".jpg")
#    im = cv2.resize(im, resize_img)
#    imgs.append(im)
#    files.append("test/IMG_0"+str(i)+".jpg")
#
#pred_imgs = np.array(np.array(imgs),'int16')
#
#for i, img in enumerate(pred_imgs):
#    fd = hog(img, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(8, 8), visualize=False)
#    nbr = clf.predict(np.array([fd], 'float64'))
#    print("{} ---> {}".format(files[i], nbr[0]))