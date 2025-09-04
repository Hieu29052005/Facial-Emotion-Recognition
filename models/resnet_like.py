import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block(x, filters, downsample=False):
    shortcut = x
    strides = (2, 2) if downsample else (1, 1)

    x = layers.Conv2D(filters, (3, 3), padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet_like(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 128, downsample=True)
    x = resnet_block(x, 256, downsample=True)
    x = resnet_block(x, 512, downsample=True)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
