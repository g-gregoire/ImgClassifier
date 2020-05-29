### broDiff - telling Antoine & Georges aprt
## This code will attempt to tell Antoine & Georges apart
## Created Nov.23, 2019
## Georges Gregoire

import os
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Image Org
base_dir = os.path.expanduser("~/Documents/code/python/broDiff/pics/")
train_dir = os.path.join(base_dir, 'Train')
test_dir = os.path.join(base_dir, 'Test')

# Directory with our training Antoine/Georges pictures
train_A_dir = os.path.join(train_dir, 'Antoine')
train_G_dir = os.path.join(train_dir, 'Georges')
# Directory with our validation Antoine/Georges pictures
test_A_dir = os.path.join(test_dir, 'Antoine')
test_G_dir = os.path.join(test_dir, 'Georges')

# Model Creation
model = tf.keras.models.Sequential([
      # Note the input shape is the desired size of the image 150x150 with 3 bytes color
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2), 
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(), 
      tf.keras.layers.Dense(512, activation='relu'), 
      tf.keras.layers.Dense(1, activation='sigmoid')]) 

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])

#Image Data Generator
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     

# Flow validation images in batches of 5 using test_datagen generator
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                         batch_size=10,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

# Step 5: Finally fit the model using .fit_generator
history = model.fit_generator(train_generator,
                              validation_data=test_generator,
                              steps_per_epoch=40,
                              epochs=25,
                              validation_steps=20,
                              verbose=2)

# # Visual Representation
# #-----------------------------------------------------------
# acc      = history.history[     'acc' ]
# val_acc  = history.history[ 'val_acc' ]
# loss     = history.history[    'loss' ]
# val_loss = history.history['val_loss' ]

# epochs   = range(len(acc)) # Get number of epochs

# #------------------------------------------------
# # Plot training and validation accuracy per epoch
# #------------------------------------------------
# plt.plot  ( epochs,     acc )
# plt.plot  ( epochs, val_acc )
# plt.title ('Training and validation accuracy')
# plt.figure()

# #------------------------------------------------
# # Plot training and validation loss per epoch
# #------------------------------------------------
# plt.plot  ( epochs,     loss )
# plt.plot  ( epochs, val_loss )
# plt.title ('Training and validation loss'   )

# predicting images
predict_dir = os.path.join(base_dir, 'Predict')
pics = os.listdir( predict_dir )

for pic in pics:
	image_str = os.path.join(predict_dir, pic)
	print(pic[0])
	if pic[0] == '.':
		print("%s is not an image file" % pic)
	else:
		# print(image_str)
		img=image.load_img(image_str, target_size=(150, 150))
	  
		x=image.img_to_array(img)
		x=np.expand_dims(x, axis=0)
		images = np.vstack([x])
		  
		classes = model.predict(images, batch_size=1)
		print(classes[0])
