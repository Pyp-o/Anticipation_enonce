from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM, Dense

def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range (length)]

def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

def generate_example(length, n_features, out_index):
	sequence = generate_sequence(length, n_features)
	encoded = one_hot_encode(sequence, n_features)
	X = encoded.reshape((1, length, n_features))
	y = encoded[out_index].reshape(1, n_features)
	return X, y


length = 5
n_features = 10
out_index = 2

model = Sequential()

model.add(LSTM(25))

model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.build(input_shape=(5, 10))
print(model.summary())

for i in range(10000):
	X, y = generate_example(length, n_features, out_index)
	#model.fit(X, y, epochs=1)

correct = 0
for i in range(100):
	X, y = generate_example(length, n_features, out_index)
	yhat = model.predict(X)
	if one_hot_decode(yhat) == one_hot_encode(y):
		correct +=1
print('Accuracy: %f', ((correct/100)*100.0))

X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence: %s', [one_hot_encode(x) for x in X])
print('Expected: %s', one_hot_decode(y))
print('Predicted: %s', one_hot_decode(yhat))