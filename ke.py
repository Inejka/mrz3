global history
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
import altair as alt
import pandas as pd
import numpy as np


def example():
    # features
    X = np.array([[[0, 0, 1, 1]],
                  [[0, 1, 0, 1]]]).T
    # expected values
    y = np.array([[0, 1, 1, 0]]).T
    print(f'training data shape: {X.shape}')
    print(f'targets data shape: {y.shape}')
    # Define a network as a linear stack of layers
    model = Sequential()
    # Add a recurrent layer with 2 units
    model.add(SimpleRNN(2, input_shape=(1, 2)))
    # Add the output layer with a sigmoid activation
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='Adadelta',
                  loss='mean_squared_error',
                  metrics=['acc'])
    history = model.fit(X, y,
                        epochs=5000,
                        verbose=0)
    alt.data_transformers.disable_max_rows()
    loss = history.history['loss']
    accuracy = history.history['acc']
    df = pd.DataFrame({"accuracy": accuracy, "loss": loss, "time-step": np.arange(0, len(accuracy))})
    base = alt.Chart(df).mark_line(color="blue").encode(x="time-step", y="accuracy")
    loss = alt.Chart(df).mark_line(color="red").encode(x="time-step", y="loss")
    (base + loss).properties(title='Chart 2').resolve_scale(y='independent')
    print(model.predict(X))


def test():
    model = Sequential()
    #X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).T
    #Y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).T
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]]).T
    Y = np.array([[2], [3], [4], [5], [6], [7], [8], [9], [10]]).T
    # Add a recurrent layer with 2 units
    model.add(SimpleRNN(units=10,input_shape=(1,1)))
    # Add the output layer with a sigmoid activation
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='Adadelta',
                  loss='mean_squared_error',
                  metrics=['acc'])
    history = model.fit(X, Y,
                        epochs=1000,
                        verbose=0)
    print(model.predict)


test()
# example()
