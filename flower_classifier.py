import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing
image_size = (224, 224)
batch_size = 32

train_data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_data.flow_from_directory(
    'flower_photos',
    target_size=image_size,
    batch_size=batch_size,
    subset='training'
)

validation_generator = train_data.flow_from_directory(
    'flower_photos',
    target_size=image_size,
    batch_size=batch_size,
    subset='validation'
)

# Model architecture
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(image_size[0], image_size[1], 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Model training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10

model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model
model.save('improved_flower_classifier.h5')
