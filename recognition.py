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

    def feature_extraction(self, dir_path, count):
        generator = self.datagen.flow_from_directory(
            dir_path,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary')

        features = np.zeros(shape=(count, 4, 4, 512))
        labels = np.zeros(shape=(count))

        i = 0
        for inputs_batch, labels_batch in generator:
            print("{}/{}".format(i * self.batch_size, count))

            features_batch = self.conv_base.predict(inputs_batch)
            features[i * self.batch_size: (i + 1) * self.batch_size] = features_batch
            labels[i * self.batch_size: (i + 1) * self.batch_size] = labels_batch

            i += 1
            if (i * self.batch_size >= count): break

        return features, labels