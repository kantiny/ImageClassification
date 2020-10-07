import tensorflow as tf
from tensorflow import keras
from matplotlib import image

import os
import numpy as np
import matplotlib.pyplot as plt
import io


batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
dirname = 'C:/Users/Аня/Desktop/Тестовое задание/ImageClassification/data/'
train_dir = os.path.join(dirname, 'IDRND_FASDB_train')
test_dir = os.path.join(dirname, 'test')


def train_predict():
    train_real_dir = os.path.join(train_dir, 'real')  # директория с картинками оригинала для обучения
    train_spoof_dir = os.path.join(train_dir, 'spoof')  # директория с картинками копий для обучения

    test_real_dir = os.path.join(train_dir, 'real')  # директория с картинками оригинала для тестирования
    test_spoof_dir = os.path.join(train_dir, 'spoof')  # директория с картинками копий для тестирования

    total_train = len(os.listdir(train_real_dir)) + len(os.listdir(train_spoof_dir))
    total_val = len(os.listdir(test_real_dir)) + len(os.listdir(test_spoof_dir))

    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)  # Генератор для тренировочных данных

    # Training
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Training
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_steps=total_val // batch_size
    )

    model.save_weights("my_model_weights.h5")

    acc = history.history['acc']
    loss = history.history['loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# train_predict()

model.load_weights('my_model_weights.h5')
with open("datatest_result.txt", mode='w') as f:
    for image in os.listdir(test_dir):
        if image[len(image)-4:len(image)] == ".png":
            im = tf.keras.preprocessing.image.load_img(os.path.join(test_dir, image), target_size=(IMG_HEIGHT, IMG_WIDTH))
            im = tf.keras.preprocessing.image.img_to_array(im)
            im = np.expand_dims(im, axis=0)
            im /= 255
            print(image + "," + str(1 - model.predict(im)[0, 0]))
            f.write(image + "," + str(1 - model.predict(im)[0, 0]) + "\n")
