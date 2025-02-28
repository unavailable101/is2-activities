from numpy import loadtxt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# for python version 3.12 kay di mu gana ang kng tensorflow.keras.models ug layers
# from keras.models import Sequential
# from keras.layers import Dense

# load data
data = loadtxt('keras simple activity/pima-indians-diabetes.csv', delimiter=',')
x = data[:, 0:8]
y = data[:, 8]

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(x)

rounded = [round(x[0]) for x in predictions]

for i in range(5):
    print('%s => %d (expected %d)' % (x[i].tolist(), rounded[i], y[i]))