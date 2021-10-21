import os
import numpy as np
import random
from PIL import Image

from .images import get_images_labels
from .pca import PCA

class OmniglotGenerator(object):
    def __init__(self, data_folder, nb_classes=5, nb_samples_per_class=10, img_size = (20, 20)):
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.img_size = img_size
        self.images = []

        # Create a PCA object
        self._pca = PCA(num_channels=30)

        for dirname, subdirname, filelist in os.walk(data_folder):
            if filelist:
                list_ = []

                for filename in filelist:
                    if filename == '.DS_Store':
                        continue

                    full_spectral = np.load(dirname + '/' + filename, allow_pickle=True)
                    self._pca.set_spectral_image(full_spectral)
                    list_.append(self._pca.perform_pca(num_channels=1))

                self.images.append(
                    list_
                )
        num_train = 400
        self.train_images = self.images[:num_train]
        self.test_images = self.images[num_train:]
        print('train_images', len(self.train_images))
        print('test_images', len(self.test_images))

    def sample(self, batch_type, batch_size, sample_strategy="random"):
        if batch_type == "train":
            data = self.train_images
        elif batch_type == "test":
            data = self.test_images

        sampled_inputs = np.zeros((batch_size, self.nb_classes * self.nb_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        sampled_outputs = np.zeros((batch_size, self.nb_classes * self.nb_samples_per_class), dtype=np.int32)

        for i in range(batch_size):
            images, labels = get_images_labels(data, self.nb_classes, self.nb_samples_per_class, self.img_size, sample_strategy)
            sampled_inputs[i] = np.asarray(images, dtype=np.float32)
            sampled_outputs[i] = np.asarray(labels, dtype=np.int32)
        return sampled_inputs, sampled_outputs






