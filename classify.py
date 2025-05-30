import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
py
def prepare_data(original_dir, base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(os.path.join(base_dir, 'train/cats'))
        os.makedirs(os.path.join(base_dir, 'train/dogs'))
        os.makedirs(os.path.join(base_dir, 'val/cats'))
        os.makedirs(os.path.join(base_dir, 'val/dogs'))

        images = os.listdir(original_dir)
        random.shuffle(images)

        split = int(0.8 * len(images))
        train_images = images[:split]
        val_images = images[split:]

        for img in train_images:
            label = 'cat' if img.startswith('cat') else 'dog'
            shutil.copy(os.path.join(original_dir, img), os.path.join(base_dir, f'train/{label}', img))

        for img in val_images:
            label = 'cat' if img.startswith('cat') else 'dog'
            shutil.copy(os.path.join(original_dir, img), os.path.join(base_dir, f'val/{label}', img))

prepare_data('data/train', 'data_split')

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data_split/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'data_split/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

model.save("cat_dog_classifier.h5")
