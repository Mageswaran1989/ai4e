import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
# The tfds.load method downloads and caches the data, and returns a tf.data.Dataset object.
# These objects provide powerful, efficient methods for manipulating data and piping it into
# your model.
import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
from tqdm import tqdm

from aie.utils.pretty_print import print_debug

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

BATCH_SIZE = 32


class ImageDatasetInfo(object):
    NAME = None
    WIDTH = None
    HEIGHT = None
    FV_SIZE = None
    IMAGE_SIZE = HEIGHT, WIDTH
    MODULE_HANDLE = None

class MobileNet(ImageDatasetInfo):
    NAME = "mobilenet"
    WIDTH = 224
    HEIGHT = 224
    FV_SIZE = 1280
    IMAGE_SIZE = HEIGHT, WIDTH
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(NAME)


class InceptionV3(ImageDatasetInfo):
    NAME = "inception_v3"
    WIDTH = 299
    HEIGHT = 299
    FV_SIZE = 2048
    IMAGE_SIZE = HEIGHT, WIDTH
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(NAME)


class FashionMNISTInfo(ImageDatasetInfo):
    NAME = "fashion_mnist"
    WIDTH = 28
    HEIGHT = 28
    IMAGE_SIZE = HEIGHT, WIDTH


class MobileNetV2(ImageDatasetInfo):
    NAME = "mobilenet_v2"
    WIDTH = 224
    HEIGHT = 224
    FV_SIZE = 1280
    IMAGE_SIZE = HEIGHT, WIDTH
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(NAME)

# =====================================================================================================================

class ImageUtils(object):
    @staticmethod
    def format_image(image, new_size, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, new_size) / 255.0
        return image, label

# =====================================================================================================================

class IDataset(object):
    def __init__(self,
                 batch_size,
                 image_data_info: ImageDatasetInfo):
        self._batch_size = batch_size
        self._image_data_info = image_data_info
        self._class_names = None

    def load_example(self):
        raise NotImplementedError

    def prepare_batches(self):

        train_examples, validation_examples, test_examples, info = self.load_examples()

        self.num_examples = info.splits['train'].num_examples
        self.num_classes = info.features['label'].num_classes

        print(self.num_examples)
        print(self.num_classes)

        self.train_batches = train_examples. \
            shuffle(self.num_examples // 4). \
            map(lambda img, lbl: ImageUtils.format_image(image=img, label=lbl,
                                                         new_size=self._image_data_info.IMAGE_SIZE)). \
            batch(self._batch_size). \
            prefetch(1)
        self.validation_batches = validation_examples. \
            map(lambda img, lbl: ImageUtils.format_image(image=img, label=lbl,
                                                         new_size=self._image_data_info.IMAGE_SIZE)). \
            batch(self._batch_size). \
            prefetch(1)
        self.test_batches = test_examples. \
            map(lambda img, lbl: ImageUtils.format_image(image=img, label=lbl,
                                                         new_size=self._image_data_info.IMAGE_SIZE)). \
            batch(1)

    def plot_image(self, i, predictions_array, true_label, img):

        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        img = np.squeeze(img)
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label: int = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self._class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self._class_names[true_label]),
                   color=color)


class CatsVsDogsDataset(IDataset):
    def __init__(self,
                 batch_size,
                 image_data_info: ImageDatasetInfo):

        IDataset.__init__(self,
                          batch_size=batch_size,
                          image_data_info=image_data_info)

        self._class_names = ['cat', 'dog']
        self.prepare_batches()

    def load_examples(self):
        # Since "cats_vs_dog" doesn't define standard splits, use the subsplit feature to divide it into
        # (train, validation, test) with 80%, 10%, 10% of the data respectively.
        (train_examples, validation_examples, test_examples), info = tfds.load(
            'cats_vs_dogs',
            split=['train[80%:]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
            data_dir="/opt/datasets/"
        )
        return train_examples, validation_examples, test_examples, info


class RockPaperScissors(IDataset):
    def __init__(self,
                 batch_size,
                 image_data_info: ImageDatasetInfo):

        IDataset.__init__(self,
                          batch_size=batch_size,
                          image_data_info=image_data_info)

        self._class_names = ['rock', 'paper', 'scissors']
        self.prepare_batches()

    def load_examples(self):
        # Since "cats_vs_dog" doesn't define standard splits, use the subsplit feature to divide it into
        # (train, validation, test) with 80%, 10%, 10% of the data respectively.
        (train_examples, validation_examples, test_examples), info = tfds.load(
            'rock_paper_scissors',
            split=['train[80%:]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
            data_dir="/opt/datasets/"
        )
        return train_examples, validation_examples, test_examples, info


# =====================================================================================================================

class FashionMNISTDataset(IDataset):
    def __init__(self,
                 batch_size,
                 image_data_info: ImageDatasetInfo):

        IDataset.__init__(self,
                          batch_size=batch_size,
                          image_data_info=image_data_info)
        self._class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.prepare_batches()

    def load_examples(self):
        splits, info = tfds.load('fashion_mnist',
                                 with_info=True,
                                 as_supervised=True,
                                 split=['train[90%:]', 'train[90%:]', 'test'],)

        (train_examples, validation_examples, test_examples) = splits
        return train_examples, validation_examples, test_examples, info


# =====================================================================================================================


class TransferLearningModel(object):
    def __init__(self,
                 dataset: CatsVsDogsDataset,
                 pre_trained_dataset_info: ImageDatasetInfo,
                 do_fine_tuning=False):
        self._dataset = dataset
        feature_extractor = hub.KerasLayer(pre_trained_dataset_info.MODULE_HANDLE,
                                           input_shape=pre_trained_dataset_info.IMAGE_SIZE + (3,),
                                           output_shape=[pre_trained_dataset_info.FV_SIZE],
                                           trainable=do_fine_tuning)
        print("Building model with", pre_trained_dataset_info.MODULE_HANDLE)
        self.model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(dataset.num_classes)
        ])
        self.model.summary()

        if do_fine_tuning:
            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(lr=0.002, momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        else:
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

#=======================================================================================================================

class FashionMNISTClassificationModel(object):
    def __init__(self,
                 dataset):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=16,
                                   input_shape=(28,28,1),
                                   activation="relu",
                                   kernel_size=3),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=32,
                                   activation="relu",
                                   kernel_size=3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,
                                  activation="relu"),
            tf.keras.layers.Dense(dataset.num_classes)
        ])

        self.model.compile(optimizer="adam",
                           # loss="softmax",
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

#=======================================================================================================================

class Trainer(object):
    def __init__(self,
                 dataset,
                 keras_model,
                 model_path_name):
        self.model = keras_model
        self._dataset = dataset

        self._model_path_name = model_path_name
        self._tflite_model_file = self._model_path_name + '_converted_model.tflite'

    def train(self, epochs=5):
        hist = self.model.fit(self._dataset.train_batches,
                              epochs=epochs,
                              validation_data=self._dataset.validation_batches)

        tf.saved_model.save(self.model, self._model_path_name)
        # saved_model_cli show --dir $1 --tag_set serve --signature_def serving_default
        loaded = tf.saved_model.load(self._model_path_name)

        print(list(loaded.signatures.keys()))
        infer = loaded.signatures["serving_default"]
        print(infer.structured_input_signature)
        print(infer.structured_outputs)

    def convert_to_lite(self):

        def representative_data_gen():
            for input_value, _ in self._dataset.test_batches.take(100):
                yield [input_value]

        converter = tf.lite.TFLiteConverter.from_saved_model(self._model_path_name)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_model = converter.convert()

        with open(self._tflite_model_file, "wb") as f:
            f.write(tflite_model)

    def tflite_intrepreter(self):
        # Load TFLite model and allocate tensors.

        interpreter = tf.lite.Interpreter(model_path=self._tflite_model_file)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Gather results for the randomly sampled test images
        predictions = []

        test_labels, test_imgs = [], []
        for img, label in tqdm(dataset.test_batches.take(10)):
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            predictions.append(interpreter.get_tensor(output_index))

            test_labels.append(label.numpy()[0])
            test_imgs.append(img)

        for index in range(10):
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            self._dataset.plot_image(index, predictions, test_labels, test_imgs)
            plt.show()
            input()


if __name__ == "__main__":

    # dataset = CatsVsDogsDataset(image_data_info=InceptionV3,
    #                             batch_size=BATCH_SIZE)

    dataset = RockPaperScissors(image_data_info=InceptionV3,
                                batch_size=BATCH_SIZE)

    model = TransferLearningModel(dataset=dataset,
                                  pre_trained_dataset_info=InceptionV3,
                                  do_fine_tuning=False)


    # dataset = FashionMNISTDataset(batch_size=BATCH_SIZE,
    #                               image_data_info=FashionMNISTInfo)
    # model = FashionMNISTClassificationModel(dataset=dataset)

    trainer = Trainer(dataset=dataset,
                      keras_model=model.model,
                      model_path_name="rock_paper_scissors_tf_modeled")

    trainer.train(epochs=5)
    trainer.convert_to_lite()
    trainer.tflite_intrepreter()