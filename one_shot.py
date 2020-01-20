import tensorflow as tf
import numpy as np
import pickle
import cv2
import sys

diff_model=tf.keras.models.load_model('diff_model.h5')
encode_model=tf.keras.models.load_model('encode_model.h5')


data=pickle.load(open('one_shot_inputs.pickle','rb'))
data=np.asarray(data)
label=np.array(data[:,1])
image_path=sys.argv[1]

def pre(img):
	inp=encode_model.predict(np.expand_dims(img,0))
	x=np.zeros((200,2048,))
	i=0
	for y in data[:,0]:
		x[i,:]=np.sqrt(np.square(y-inp[0]))
		i=i+1
	return x

def prediction(x):
	y=diff_model.predict(x)
	a=None
	n=np.argmax(y)
	if y[n]>0.7:
		a=label[n]
	return a

def frames(frame):
	img=cv2.resize(frame,(299,299))
	x=pre(img)
	y=prediction(x)
	image = cv2.putText(frame, y, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
	return image

cap = cv2.VideoCapture(image_path)
ret=True 
while(ret):
    ret, frame = cap.read()
    try:
    	image = frames(frame)
    except:
    	print("Failed")
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()