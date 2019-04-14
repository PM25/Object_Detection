from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class Recognition:
    def __init__(self, batch_size=20):
        self.conv_base = VGG16(weights="imagenet",
                               include_top=False,
                               input_shape=(150, 150, 3))
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.model = self.make_model()

    def feature_extraction(self, dir_path, samp_count):
        self.generator = self.datagen.flow_from_directory(
            dir_path,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary')

        features = np.zeros(shape=(samp_count, 4, 4, 512))
        labels = np.zeros(shape=(samp_count))

        i = 0
        for inputs_batch, labels_batch in self.generator:
            print("{}/{}".format(i * self.batch_size, samp_count))

            features_batch = self.conv_base.predict(inputs_batch)
            features[i * self.batch_size: (i + 1) * self.batch_size] = features_batch
            labels[i * self.batch_size: (i + 1) * self.batch_size] = labels_batch

            i += 1
            if (i * self.batch_size >= samp_count): break

        return features, labels

    def show_img(self):
        pass

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['acc'])

        return model

    def fit_model(self, train_features, train_labels,
                  validation_features, validation_labels,
                  epochs=30,
                  batch_size=20):
        train_shape = np.shape(train_features)
        train_shape = (train_shape[0], np.prod(train_shape[1:]))
        train_features = np.reshape(train_features, train_shape)

        validation_shape = np.shape(validation_features)
        validation_shape = (validation_shape[0], np.prod(validation_shape[1:]))
        validation_features = np.reshape(validation_features, validation_shape)

        history = self.model.fit(train_features, train_labels,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(validation_features, validation_labels))

        return history