from PIL import Image
import numpy as np
import pickle
import os

use_first_batch_only = True
_num_test_images_to_use = 1000


########################################################################

# Directory where our CIFAR-10 data is at
data_path = "../data/"


########################################################################
# Various constants for the size of the images.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
if use_first_batch_only:
    _num_files_train = 1
# using only the first batch for training
else:
    _num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
def showImage(image_array, title=None):
    '''
    :param image_array: a numpy array
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'))
    im.show(title="abc")


def saveImage(image_array, name, loc = None):
    '''
    saves the image array 'image_array' as image at the given location 'loc' with the given name
    :param image_array:
    :param name:
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'))
    im.save(name)


def _get_file_path(filename=""):
    '''
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    :param filename:
    :return:
    '''

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    '''
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    :param filename:
    :return:
    '''

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    '''
    convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0 and 255
    :param raw:
    :return:
    '''

    # Convert the raw images from the data-files to floating-points.
    # raw_float = np.array(raw, dtype=float) / 255.0
    raw_float = np.array(raw, dtype=float)

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    '''
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    :param filename:
    :return:
    '''

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls



def load_class_names():
    '''
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    :return:
    '''

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data(first_batch_only = False):
    '''
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers
    :param first_batch_only:
    :return:
    '''

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls


def load_test_data():
    '''
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers
    :return:
    '''

    images, cls = _load_data(filename="test_batch")

    return images[:_num_test_images_to_use], cls[:_num_test_images_to_use]

########################################################################

if __name__ == '__main__':
    images, cls = load_training_data(first_batch_only=True)
    images = np.average(images, axis=3)