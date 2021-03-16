import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


NCHANNELS = 2
NCLASSES = 11
SAMPLE_SHAPE = (128,259)
def make_model():
	"""
	Creates a list of NCLASSES binary classification models, each one identical
	"""
	models = []
	for i in range(NCLASSES):
		model = tf.keras.Sequential()

		model.add(tf.keras.layers.Conv2D(trainable=False, filters=8, kernel_size=4, strides=(2,2), padding='same', input_shape=(128,259,NCHANNELS), name="conv_1"))
		model.add(tf.keras.layers.BatchNormalization(name="bn_1"))
		model.add(tf.keras.layers.ReLU())
		model.add(tf.keras.layers.MaxPooling2D(trainable=False, pool_size=(2,2), name="mpool_1"))

		model.add(tf.keras.layers.Conv2D(trainable=False, filters=16, kernel_size=3, strides=(1,1), padding='valid', name="conv_2"))
		model.add(tf.keras.layers.BatchNormalization(name="bn_2"))
		model.add(tf.keras.layers.ReLU())
		model.add(tf.keras.layers.MaxPooling2D(trainable=False, pool_size=(2,2), name="mpool_2"))

		model.add(tf.keras.layers.Conv2D(trainable=False, filters=32, kernel_size=2, strides=(1,1), padding='valid', name="conv_3"))
		model.add(tf.keras.layers.BatchNormalization(name="bn_3"))
		model.add(tf.keras.layers.ReLU())
		model.add(tf.keras.layers.MaxPooling2D(trainable=False, pool_size=(2,2), name="mpool_3"))

		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(500, trainable=False, activation='relu', name="fc_4"))
		model.add(tf.keras.layers.Dense(NCLASSES, trainable=False, activation='relu', name="fc_5"))
		model.add(tf.keras.layers.Dense(1, activation='sigmoid', name="fc_6"))

		model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
		models.append(model)

	return models

def process_dataframe(dataframe):
	"""
	Convert the pandas dataframe into numpy arrays for training
	broken out so that the jupyter notebook doesnt have to do this every flipping time i need to edit the model
	"""
	x, dataframe_y = np.stack(dataframe.data), np.stack(dataframe.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))
	ys = []
	for i in range(NCLASSES):
		ys.append(get_y_labels(dataframe_y, i))

	return x, ys

def train_model(models, x, y, batch_size=32, epochs=15):
	"""
	train the NCLASSES binarry classifiers
	"""
	history = []
	for i in range(NCLASSES):
		print("Training model:", i)
		new_history = models[i].fit(x, y[i], batch_size=batch_size, epochs=epochs, class_weight=get_class_weights(i))
		history.append (new_history.history)
	return history


def get_y_labels(dataframe_y, selected_class):
	"""
	Takes in the multiclass y labels and converts it into a set of single class labels
	such that the label is 1 if the selected class was 1 and 0 otherwise
	"""
	return dataframe_y[:,selected_class,0]

def get_class_weights(selected_class):
	"""
	Upweight the positive class, since for all 11 models we will have 10 negative examples for every 1 positive
	"""
	return {0:1.0, 1:10.0}

"""
def get_accuracy(models, dataset):
	x, y = np.stack(dataset.data), np.stack(dataset.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))

	preds = [models[i].predict(x) for i in range(NCLASSES)]
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
"""