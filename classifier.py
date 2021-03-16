import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

NCHANNELS = 2
NCLASSES = 11
SAMPLE_SHAPE = (128,259)
def make_model():
	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=4, strides=(2,2), padding='same', input_shape=(128,259,NCHANNELS), name="conv_1"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_1"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_1"))

	model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding='valid', name="conv_2"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_2"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_2"))

	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1,1), padding='valid', name="conv_3"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_3"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_3"))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(500, activation='relu', name="fc_4"))
	model.add(tf.keras.layers.Dense(NCLASSES, activation=None, name="fc_5"))
	model.add(tf.keras.layers.Softmax())

	model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())

	return model


def train_model(model, train_set, batch_size=32, epochs=15):
	x, y = np.stack(train_set.data), np.stack(train_set.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))
	return model.fit(x, y, batch_size=batch_size, epochs=epochs)


def get_accuracy(model, dataset):
	x, y = np.stack(dataset.data), np.stack(dataset.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))

	preds = np.argmax(model.predict(x), axis=1).reshape(x.shape[0], 1)
	true = np.argmax(y, axis=1)
	acc = np.sum((preds == true)*np.ones(preds.shape))/x.shape[0]

	return acc


def predict_song(model, song_data, stride=50):
	#"convolves" the model along an entire song
	#better to have the stride bigger, 259 samples is 3 seconds or so
	length = song_data.shape[1]
	n_iters = int((length - SAMPLE_SHAPE[1])/stride)
	output = np.zeros((n_iters, NCLASSES))
	for i in range(n_iters):
		segment = song_data[:,i*stride:i*stride + SAMPLE_SHAPE[1],:]
		segment = segment.reshape((1, SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], NCHANNELS))
		output[i] = model.predict(segment)

	return output
