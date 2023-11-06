from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import warnings
def test_func():
    print("okie model start now uwu")
    warnings.simplefilter("ignore")

    df = pd.read_csv("./content/station_day.csv")
    df1= df.dropna()
    lstm_df = pd.read_excel('./content/cancer patient data sets.xlsx')

    def remove_outliers(df1, column_name):
        Q1 = df1['AQI'].quantile(0.25)
        Q3 = df1['AQI'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df1[(df1['AQI'] >= lower_bound) & (df1['AQI'] <= upper_bound)]

    df1 = remove_outliers(df1, 'AQI')

    arima_df = df[['Date','AQI']]


    arima_df["Date"]= pd.to_datetime(arima_df["Date"])


    arima_df.AQI = arima_df.groupby(pd.PeriodIndex(arima_df['Date'], freq="M"))['AQI'].apply(lambda x: x.fillna(x.mean()))


    ts = arima_df.groupby(pd.PeriodIndex(arima_df['Date'], freq="M"))['AQI'].mean()


    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from numpy import log
    result = adfuller(ts)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


    ts_train = ts[:50]
    ts_test = ts[50:]


    from statsmodels.tsa.arima.model import ARIMA


    import itertools
    from sklearn.metrics import mean_squared_error


    p = range(0,8)
    q = range(0,8)
    d = range(0,2)


    pqd_combination = list(itertools.product(p,d,q))


    error = []
    pqd = []


    for i in pqd_combination:
        A_model = ARIMA(ts_train,order= i).fit()
        predict = A_model.predict(len(ts_train),len(ts)-1)
        e = np.sqrt(mean_squared_error(ts_test,predict))
        pqd.append(i)
        error.append(e)


    min = error[0]
    index = 0
    for i in range(1,len(error)-1):
        if(min > error[i]):
            min = error[i]
            index = i

    print(error[index],' => ',pqd[index])


    model_ts = ARIMA(ts_train, order=pqd[index])
    model_ts_fit = model_ts.fit()
    print(model_ts_fit.summary())


    arima_predict = model_ts_fit.predict(start = len(ts_train),end = len(ts))


    # import matplotlib.pyplot as plt
    residuals = model_ts_fit.resid[1:]
    # fig, ax = plt.subplots(1,2)
    # residuals.plot(title='Residuals', ax=ax[0])
    # residuals.plot(title='Density', kind='kde', ax=ax[1])
    # plt.show()


    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    forecast_test = model_ts_fit.forecast(len(ts_test))

    rmse = np.sqrt(mean_squared_error(ts_test, forecast_test))
    mae = mean_absolute_error(ts_test, forecast_test)
    mape = mean_absolute_percentage_error(ts_test, forecast_test)
    print(f'mae -: {mae}')
    print(f'mape -: {mape}')
    print(f'rmse -: {rmse}')



    # plt.figure(figsize = (20,10))
    # ts_test.plot(label = "Test")
    # ts_train.plot(label = "Train")
    # predict.plot(label = 'Predict')
    # plt.legend()
    # plt.show()


    # # ------------------------------- SARIM -------------------------------


    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput )

    # *****This time series is not stationary. Because P > 0.05 .So we want to take first differntioal for series*****


    df1_ts = ts - ts.shift(1)


    df1_ts=df1_ts.dropna()


    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df1_ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput )


    # plt.figure(figsize = (20,10))
    # df1_ts.plot()
    # plt.show()


    df1_ts_train = df1_ts[:50]
    df1_ts_test = df1_ts[50:]


    df_pdq = []
    df_error = []


    for i in pqd_combination:
        A_model = ARIMA(df1_ts_train,order= i).fit()
        predict = A_model.predict(len(df1_ts_train),len(df1_ts)-1)
        e = np.sqrt(mean_squared_error(df1_ts_test,predict))
        df_pdq.append(i)
        df_error.append(e)


    min = df_error[0]
    index = 0
    for i in range(1,len(df_error)-1):
        if(min > df_error[i]):
            min = df_error[i]
            index = i

    print(df_error[index],' => ',df_pdq[index])


    import statsmodels.api as sm


    sarima_model = sm.tsa.statespace.SARIMAX(df1_ts_train, trend='n', order=(7,0,3), seasonal_order=(1,1,1,12))
    s_results = sarima_model.fit()
    print(s_results.summary())


    forecast_steps = len(df1_ts_test)
    forecast = s_results.get_forecast(steps=forecast_steps)
    predicted_values = forecast.predicted_mean


    from math import sqrt

    mse = mean_squared_error(df1_ts_test, predicted_values)
    rmse = sqrt(mse)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = mean_absolute_percentage_error(df1_ts_test, predicted_values)

    # Print the metrics
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')



    s_pred = s_results.predict(start = len(df1_ts_train),end = len(df1_ts)-1)


    # plt.figure(figsize = (20,10))
    # df1_ts_test.plot(label = "Test")
    # df1_ts_train.plot(label = "Train")
    # s_pred.plot(label = 'Predict')
    # plt.legend()
    # plt.show()

    # # ---------------------------------------- LSTM ----------------------------------------


    lstm_df= pd.read_excel('./content/cancer patient data sets.xlsx')


    lstm_df.head()


    lstm_df=lstm_df.dropna()


    lstm_df.info()


    import math
    dataset  = lstm_df.values
    training_data_len = math.ceil(len(dataset)*.8)
    training_data_len


    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0,1))
    scaled_data =  sc.fit_transform(lstm_df[['Age', 'AQI']])
    scaled_data


    ################3
    scaler = MinMaxScaler()
    lstm_df[['Age', 'AQI']] = scaler.fit_transform(lstm_df[['Age', 'AQI']])


    from sklearn.model_selection import train_test_split


    X = lstm_df[['Age', 'Gender', 'AQI', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
            'chronic Lung Disease', 'Smoking', 'Passive Smoker', 'Clubbing of Finger Nails', 'Frequent Cold']]
    y = lstm_df['Level']

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    num = 60
    for i in range(num, len(train_data)):
        x_train.append(train_data[i-num:i , 0])
        y_train.append(train_data[i , 0])


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    x_train.shape



    #model = Sequential([
    #  LSTM(units=64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    #   Dense(1, activation='sigmoid')  # Assuming binary classification
    #])
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import InputLayer

    model_lstm = Sequential()

    model_lstm.add(InputLayer((12,1)))

    model_lstm.add(LSTM(50))

    model_lstm.add(Dense(34 ,'relu'))
    # model_lstm.add(Dropout(0.25))

    model_lstm.add(Dense(15 ,'relu'))

    model_lstm.add(Dense(1 ,'relu' ))


    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))



    model.compile(optimizer = "adam", loss = "mean_squared_error", metrics=['mae'])



    model.fit(x_train,y_train, batch_size=1, epochs=1)



    model_lstm.summary()


    test_data = scaled_data[training_data_len-60: , :]
    x_test = []
    y_test = dataset[training_data_len:,:]
    for i in range(num, len(test_data)):
        x_test.append(test_data[i-num:i, 0])


    x_test = np.array(x_test)
    print(x_test.shape)


    print(y_test.shape)



    y_test = np.random.rand(100, 2)
    y_test = np.concatenate((y_test, np.zeros((100, 2))), axis=0)



    # print(x_test)


    # print(y_test)

    print("Now we do za predict")


    #predictions = model.predict(x_test)
    #predictions = predictions.reshape(-1,2)
    #predictions = sc.inverse_transform(predictions)
    #predictions= predictions.squeeze()

    # Assuming x_test has shape (batch_size, 6, 10)
    #x_test = x_test.reshape(200, 11, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 60, 1)




    predictions = model.predict(x_test_reshaped)
    predictions = predictions.reshape(-1,2)
    predictions = sc.inverse_transform(predictions)


    print(predictions)


    low_threshold = 1.10  # Adjust this value based on your specific problem
    high_threshold = 1.14  # Adjust this value based on your specific problem

    thatarray = []

    # Create an empty list to store the categories
    categories = []

    # Categorize the predictions
    for prediction in predictions:
        per_cent = prediction[0] / (prediction[0]+prediction[1])
        thatarray.append(per_cent)
    x = thatarray
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    print(x_norm)

    print("model is za done :)")
    return [x_norm,"issa done"]