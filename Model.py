import preprocess1
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
import tensorflow as tf
import preprocess2
from keras.callbacks import EarlyStopping

x_train = tf.stack(preprocess2.x_train)

y_train = preprocess1.y_train[:2500]

x_train = x_train/255

model = Sequential()

model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(10, kernel_size=(2, 2), activation='relu'))
model.add(MaxPool2D([3, 3]))
model.add(Dropout(0.2))

model.add(Conv2D(16, kernel_size=(2, 2), activation='relu'))
model.add(MaxPool2D([3, 3]))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(768, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(290, activation='softmax'))

early_stop = EarlyStopping(monitor='loss', patience=4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=50, callbacks=[early_stop])

model.save('FN_Model_v1.h5')
