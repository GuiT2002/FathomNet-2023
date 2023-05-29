import os
import tensorflow as tf


def preprocess(folder_path):
    extensions = (".jpg", ".jpeg", ".png")
    # Create a list with the file paths of all images_train in the folder
    image_paths = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(extensions)])

    # Load each image into a tensor
    i = 0
    x_train = []

    for path in image_paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [152, 270])
        x_train.append(image)
        i += 1
        print(i)

    x_train = tf.stack(x_train)
    x_train = x_train/255

    return x_train
