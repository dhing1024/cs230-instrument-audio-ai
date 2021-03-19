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
	Creates a multi-classifier model
	"""

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Conv2D(filters=8, trainable=True, kernel_size=4, strides=(2,2), padding='same', input_shape=(128,259,NCHANNELS), name="conv_1"))
	model.add(tf.keras.layers.BatchNormalization(trainable=True, name="bn_1"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), trainable=True, name="mpool_1"))

	model.add(tf.keras.layers.Conv2D(filters=16, trainable=True, kernel_size=3, strides=(1,1), padding='valid', name="conv_2"))
	model.add(tf.keras.layers.BatchNormalization(trainable=True, name="bn_2"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), trainable=True, name="mpool_2"))

	model.add(tf.keras.layers.Conv2D(filters=32, trainable=True, kernel_size=2, strides=(1,1), padding='valid', name="conv_3"))
	model.add(tf.keras.layers.BatchNormalization(trainable=True, name="bn_3"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), trainable=True, name="mpool_3"))

	model.add(tf.keras.layers.Conv2D(filters=64, trainable=True, kernel_size=2, strides=(1,1), padding='valid', name="conv_4"))
	model.add(tf.keras.layers.BatchNormalization(trainable=True, name="bn_4"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), trainable=True, name="mpool_4"))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dropout(rate = 0.25))
	model.add(tf.keras.layers.Dense(500, trainable=True, activation='relu', kernel_regularizer='l2', name="fc_5"))
	model.add(tf.keras.layers.Dropout(rate = 0.5))
	model.add(tf.keras.layers.Dense(NCLASSES, trainable=True, activation=None, kernel_regularizer='l2', name="fc_6"))

	model.add(tf.keras.layers.Dense(NCLASSES, activation='relu', name="fc_new_1"))
	model.add(tf.keras.layers.Dense(NCLASSES, activation='relu', name="fc_new_2"))
	model.add(tf.keras.layers.Dense(NCLASSES, activation='sigmoid', name="fc_new_3"))

	model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())

	return model

def process_dataframe(dataframe):
	"""
	Convert the pandas dataframe into numpy arrays for training
	broken out so that the jupyter notebook doesnt have to do this every flipping time i need to edit the model
	"""
	x, y = np.stack(dataframe.data), np.stack(dataframe.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))

	return x, y

def train_model(model, train_set, batch_size=32, epochs=15):
	"""
	train the multi-classifier model
	"""
	x, y = np.stack(train_set.data), np.stack(train_set.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))
	model.fit(x, y, batch_size=batch_size, epochs=epochs)


def intersection_over_union(vec1, vec2):
	"""
	takes in two vectors containing only 0's or 1's
	returns the number of categories in which both vectors have a 1 divided by the total number of categories that have a 1 in either vector
	"""
	vec1 = np.squeeze(vec1)
	vec2 = np.squeeze(vec2)
	assert(vec1.shape == vec2.shape)

	union = np.sum(vec1 + vec2 > 0)
	intersection = np.sum(vec1 + vec2 > 1)

	return intersection/union

def get_accuracy(model, dataset):
	"""
	calculates IoU accuracy for the model
	can be used on the validation set or the test set
	"""
	x, y = np.stack(dataset.data), np.stack(dataset.label)
	x = x.reshape((-1,SAMPLE_SHAPE[0],SAMPLE_SHAPE[1],2))
	y = (y > 0.5)*1

	preds = model.predict(x)
	acc = [intersection_over_union(preds[i], y[i]) for i in range(len(preds))]
	acc = np.mean(acc)

	return acc


def load_weights(model, weighted_model):
	"""
	loads weights by layer name from a previously trained model into a new model
	will pass over layers that don't have matching names
	"""
	for layer in weighted_model.layers:
		for v in range(len(model.layers)):
			if model.layers[v].name == layer.name:
				model.layers[v].set_weights(layer.get_weights())