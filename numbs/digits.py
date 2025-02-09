import os

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization  # noqa
from tensorflow.keras.models import Sequential, load_model  # noqa
from tensorflow.keras.optimizers import Adam  # noqa

MODEL_PATH = "mnist_digit_recognizer.h5"


def create_model():
    model_ = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model_.compile(
        optimizer='adam',  # Оптимизатор Adam
        loss='sparse_categorical_crossentropy',  # Функция потерь
        metrics=['accuracy']
    )
    return model_


if os.path.exists(MODEL_PATH):
    print("Загружается обученная модель...")
    model = load_model(MODEL_PATH)
else:
    print("Модель не найдена. Начинается обучение...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255
    model = create_model()
    model.fit(
        x_train, y_train,  # Обучающие данные
        epochs=10,  # Число эпох
        validation_data=(x_test, y_test)
    )  # Валидация на тестовых
    model.save(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")