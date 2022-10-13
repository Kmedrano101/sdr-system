import numpy as np
import cv2 as cv
import pickle, random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


#from matplotlib import pyplot as plt
data = []

file = open('data/dataset/data_v1.pkl','rb')
data =  pickle.load(file)
file.close()

random.shuffle(data)

X = []
y = []

for f,l in data:
    X.append(f)
    y.append(l)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model = SVC(C=1,kernel='poly',gamma='auto')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

exactitud = accuracy_score(y_test, y_pred)
print(exactitud)

"""img = cv.imread('test1.jpg',0) # `<opencv_root>/samples/data/blox.jpg`

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imwrite('fast_true1.png', img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imwrite('fast_false1.png', img3)"""