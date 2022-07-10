# Imports
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import load_model

# keras imports
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


# We are using Haar Cascade Model trained to detect faces in an image.
face_model = cv2.CascadeClassifier('Data and Dependencies/haarcascade/haarcascade_frontalface_default.xml')

#0 = Mask; 1= No mask
#0 = Green; 1= Red
mask_label = {0:'MASK',1:'NO MASK'}
color_label = {0:(0,255,0),1:(255,0,0)}




def CreateModel(face_model):

	# Load train and test set
	train_dir = 'Data and Dependencies/Face Mask Dataset/Train'
	test_dir = 'Data and Dependencies/Face Mask Dataset/Test'
	val_dir = 'Data and Dependencies/Face Mask Dataset/Validation'

	# Data augmentation
	train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
	train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

	val_datagen = ImageDataGenerator(rescale=1.0/255)
	val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

	test_datagen = ImageDataGenerator(rescale=1.0/255)
	test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)


	# VGG-19 convolutional neural network (19 layers deep)
	vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

	for layer in vgg19.layers:
		layer.trainable = False
	    
	# Building NN model architecture
	model = Sequential()
	model.add(vgg19)
	model.add(Flatten())
	model.add(Dense(2,activation='sigmoid'))
	model.summary()

	#Compile model
	model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")

	# Train model with augmented data
	history = model.fit_generator(generator=train_generator,steps_per_epoch=len(train_generator)//32,epochs=20,validation_data=val_generator,validation_steps=len(val_generator)//32)

	# Evaluate model
	model.evaluate_generator(test_generator)
	#98% accuracy on test data.

	return model



def pipeline(face_model, model, img):
	faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)
	img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
	new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
	for i in range(len(faces)):
		(x,y,w,h) = faces[i]
		crop = new_img[y:y+h,x:x+w]
		crop = cv2.resize(crop,(128,128))
		crop = np.reshape(crop,[1,128,128,3])/255.0
		mask_result = model.predict(crop)
		cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_label[mask_result.argmax()],2)
		cv2.rectangle(new_img,(x,y),(x+w,y+h),color_label[mask_result.argmax()],1)
	return new_img





def save_webcam(outPath, fps, face_model, model, mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)

    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)


            frame = pipeline(face_model, model, frame)

            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main(face_model, model):
    save_webcam('output.avi', 30.0, face_model, model, mirror=True)



#model = CreateModel(face_model)
#model.save('masknet.h5')

model = load_model('masknet.h5')


if __name__ == '__main__':
    main(face_model,model)