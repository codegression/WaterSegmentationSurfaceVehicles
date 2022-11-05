#!/usr/bin/env python
"""
Author: Codegression
This module defines the main deep learning architecture
"""

import tensorflow as tf

#constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNEL = 3
RANDOM_SEED = 72


def create():
        """Creates a deep learning model

        Returns:
                tf.keras.Model: the main model
        """
        resnet = tf.keras.applications.ResNet50(weights='imagenet')
        #print(len(resnet.layers))
        for layer in resnet.layers:
            layer.trainable = False

        inputlayer = resnet.input
        skip1 = resnet.get_layer("conv1_relu").output #112, 112,64
        skip2 = resnet.get_layer("conv2_block3_2_relu").output #56, 56, 64
        skip3 = resnet.get_layer("conv3_block4_2_relu").output #28, 28, 128
        skip4 = resnet.get_layer("conv4_block6_out").output #14, 14, 1024
        skip5 = resnet.get_layer("conv5_block3_out").output #7, 7, 2048

        skip1 = tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU',
                                        kernel_initializer='he_normal', padding='same')(skip1)
        skip1 = tf.keras.layers.Dropout(0.2)(skip1)   #112, 112, 32

        skip4 = tf.keras.layers.Conv2D(256, (3, 3), activation='LeakyReLU',
                                        kernel_initializer='he_normal', padding='same')(skip4)
        skip4 = tf.keras.layers.Dropout(0.2)(skip4)   #14, 14, 256


        skip5 = tf.keras.layers.Conv2D(512, (3, 3), activation='LeakyReLU',
                                        kernel_initializer='he_normal', padding='same')(skip5)
        skip5 = tf.keras.layers.Dropout(0.2)(skip5)   #7, 7, 512

        #reconstruction layers

        #node5
        node5 = tf.keras.layers.Conv2DTranspose(
                256, (2, 2), strides=(2, 2), padding='same')(skip5)
        node5 = tf.keras.layers.concatenate([skip4, node5])
        node5 = tf.keras.layers.Conv2D(256, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node5)
        node5 = tf.keras.layers.Dropout(0.2)(node5)
        node5 = tf.keras.layers.Conv2D(256, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node5)


        #node4
        node4 = tf.keras.layers.Conv2DTranspose(
                128, (2, 2), strides=(2, 2), padding='same')(node5)
        node4 = tf.keras.layers.concatenate([skip3, node4])
        node4 = tf.keras.layers.Conv2D(128, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node4)
        node4 = tf.keras.layers.Dropout(0.2)(node4)
        node4 = tf.keras.layers.Conv2D(128, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node4)



        #node3
        node3 = tf.keras.layers.Conv2DTranspose(
                64, (2, 2), strides=(2, 2), padding='same')(node4)
        node3 = tf.keras.layers.concatenate([skip2, node3])
        node3 = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node3)
        node3 = tf.keras.layers.Dropout(0.2)(node3)
        node3 = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node3)

        #node2
        node2 = tf.keras.layers.Conv2DTranspose(
                32, (2, 2), strides=(2, 2), padding='same')(node3)
        node2 = tf.keras.layers.concatenate([skip1, node2])
        node2 = tf.keras.layers.Conv2D(16, (3, 3), activation='LeakyReLU',
                kernel_initializer='he_normal', padding='same')(node2)
        node2 = tf.keras.layers.Conv2DTranspose(
                1, (2, 2), strides=(2, 2), padding='same')(node2)   
        node1 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid',
                kernel_initializer='he_normal', padding='same')(node2)



        model = tf.keras.Model(inputs=[inputlayer], outputs=[node1])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['accuracy'])
                        #metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        return model

if __name__ == '__main__':
        create()