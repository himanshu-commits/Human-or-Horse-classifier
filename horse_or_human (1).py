
import os

train_horse_dir = os.path.join('/content/drive/MyDrive/colab_notebook/horse-or-human/train/horses')
train_human_dir = os.path.join('/content/drive/MyDrive/colab_notebook/horse-or-human/train/humans')
validation_horse = os.path.join('/content/drive/MyDrive/colab_notebook/horse-or-human/validation/horses')
validation_human = os.path.join('/content/drive/MyDrive/colab_notebook/horse-or-human/validation/humans')

train_horse_name = os.listdir(train_horse_dir)
print(train_horse_name[:10])
train_human_name = os.listdir(train_human_dir)
print(train_human_name[:10])
validation_horse_name = os.listdir(validation_horse)
print(validation_horse_name[:10])
validation_human_name = os.listdir(validation_human) 
print(validation_human_name[:10])

print('total training horse images', len(os.listdir(train_horse_dir)))
print('total training human images', len(os.listdir(train_human_dir)))
print('total vaidation horse images', len(os.listdir(validation_horse)))
print('total vaidation human images', len(os.listdir(validation_human)))

import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagenerator = ImageDataGenerator(rescale=1./255)
train_generator = train_datagenerator.flow_from_directory(
    '/content/drive/MyDrive/colab_notebook/horse-or-human/train',
    target_size=(300,300),
    batch_size = 128,
    class_mode = 'binary')
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '/content/drive/MyDrive/colab_notebook/horse-or-human/validation',
    target_size = (300,300),
    batch_size = 32,
    class_mode = 'binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data = validation_generator,
    validation_steps = 8
)

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()
for fn in uploaded.keys():
  path = '/content/'+fn
  img = image.load_img(path,target_size = (300,300))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis = 0)

  images = np.vstack([x])
  classes = model.predict(images,batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")

