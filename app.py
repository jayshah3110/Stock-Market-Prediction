import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import streamlit as st

# Download stock data
start = '2020-01-01'
end = '2023-12-31'
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Data exploration
st.subheader('Data from 2020 - 2023')
st.write(df.describe())

# Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Closing Price vs Time Chart with 100MA
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

# Closing Price vs Time Chart with 100MA & 200MA
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'])
st.pyplot(fig)

# Data preprocessing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = StandardScaler()
data_training_array = scaler.fit_transform(data_training)

# Prepare data for training
x_train, y_train = [], []
for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i, 0])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Model creation and training
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Save the model
model.save('stock_prediction_model.h5')

# Load the model
model = load_model('stock_prediction_model.h5')

# Prepare data for testing
data = pd.concat([data_training, data_testing], ignore_index=True)
inputs = data[len(data) - len(data_testing) - 100:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(100, len(inputs)):
    x_test.append(inputs[i-100:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plotting predictions vs original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(predictions):], data_testing.values, label='Original Price')
plt.plot(df.index[-len(predictions):], predictions, label='Predicted Price', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
