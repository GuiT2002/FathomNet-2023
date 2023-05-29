import keras.models
import tensorflow as tf
import preprocess2
import preprocess1
from keras.callbacks import EarlyStopping

model = keras.models.load_model('FN_Model_v1')

x_train = tf.stack(preprocess2.x_train)

y_train = preprocess1.y_train[2500:]

x_train = x_train/255

early_stop = EarlyStopping(monitor='val_loss', patience=6)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.15, callbacks=[early_stop])

model.save('FN_Model_v1A.h5')
