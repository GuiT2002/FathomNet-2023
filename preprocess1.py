import os
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
import numpy as np

folder_path = '/images_train'
extensions = (".jpg", ".jpeg", ".png")
csv_path = "C:\\Users\\huilh\\OneDrive\\Área de Trabalho\\AI Training Models\\Kaggle Competitions\\FathomNet " \
           "2023\\multilabel_classification\\train.csv"

images_names = [file for file in os.listdir(folder_path) if file.endswith(extensions)]
images_names = [filename[:-4] for filename in images_names]


with open(csv_path, 'r') as f:
    df = pd.read_csv(f)

# Filter the rows in the dataframe where the id column matches one of the image names
filtered_df = df[df['id'].isin(images_names)]

# Extract the categories values from the filtered dataframe
labels = filtered_df['categories'].tolist()
labels = [eval(label) for label in labels]

# Find the maximum number of classes
max_classes = max(len(label) for label in labels)

# Pad the labels with zeros
padded_labels = pad_sequences(labels, maxlen=max_classes, padding='post')

# Convert the labels to integer-encoded format
int_labels = [np.argmax(label) for label in labels]

# Convert the integer-encoded labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(int_labels, num_classes=290)

print(tf.shape(y_train))