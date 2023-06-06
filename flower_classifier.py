import tensorflow as tf
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Rescaling
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib
import PIL.Image


# # Load the data
image_height = 180
image_width = 180
batch_size = 32
data_dir = pathlib.Path("flower_photos")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)

#analysing the data
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
PIL.Image.open(str(roses[0])).show()

# #visualising the data
class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off") 
plt.show()

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#create model

# model = Sequential()
# model.add(Rescaling(1./255))
# model.add(Conv2D(32,3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32,3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32,3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))


num_classes = 5
filters = 32  # @param
kernel_size = [3, 3]  # @param
strides = 1 # @param
hidden_units = 128  # @param
dropout_rate = 0.2  # @param
learning_rate = 0.01 # @param

model = tf.keras.Sequential([
  tf.keras.Input(shape=(image_height, image_width, 3)),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip('horizontal'),
  layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
  layers.Conv2D(filters, kernel_size, strides, padding='same', activation='relu'),
  layers.MaxPooling2D(pool_size=[2, 2], strides=2),
  layers.Conv2D(2 * filters, kernel_size, strides, padding='same', activation='relu'),
  layers.MaxPooling2D(pool_size=[2, 2], strides=2),
  layers.Conv2D(4 * filters, kernel_size, strides, padding='same', activation='relu'),
  layers.Conv2D(4 * filters, kernel_size, strides, padding='same', activation='relu'),
  layers.MaxPooling2D(pool_size=[2, 2], strides=2),
  layers.Flatten(),
  layers.Dropout(rate=dropout_rate),
  layers.Dense(hidden_units, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])


model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
epochs = 30
history = model.fit(train_ds,validation_data = val_ds,epochs=epochs)

model.summary()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#save model
model.save("flower_classifier.h5")

#load model
model = tf.keras.models.load_model("flower_classifier.h5")

#predict
img = cv2.imread("<image path with image name included>")
img = cv2.resize(img,(180,180))
img = np.reshape(img,[1,180,180,3])
classes = model.predict(img)
print(classes)
print(f'this is a {class_names[np.argmax(classes)]}')
