import pandas as pd
import numpy as np
def arima():
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

    print(pqd_combination)
    # exit()


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


    import matplotlib.pyplot as plt
    residuals = model_ts_fit.resid[1:]
    fig, ax = plt.subplots(1,2)
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    plt.show()


    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    forecast_test = model_ts_fit.forecast(len(ts_test))

    rmse = np.sqrt(mean_squared_error(ts_test, forecast_test))
    mae = mean_absolute_error(ts_test, forecast_test)
    mape = mean_absolute_percentage_error(ts_test, forecast_test)
    print(f'mae -: {mae}')
    print(f'mape -: {mape}')
    print(f'rmse -: {rmse}')



    plt.figure(figsize = (20,10))
    ts_test.plot(label = "Test")
    ts_train.plot(label = "Train")
    predict.plot(label = 'Predict')
    plt.legend()
    plt.show()

    # Assume avg_aqi is the average AQI for a month
    avg_aqi = 150

    # Convert the average AQI to a pandas Series
    new_data = pd.Series([avg_aqi])

    # Use the trained model to make a forecast for the next month
    forecast_next_month = model_ts_fit.forecast(steps=1, exog=new_data)

    # The forecast for the next month's AQI is the first element of the forecast
    next_month_aqi = forecast_next_month[0][0]

    print(f'The forecast for the next month\'s AQI is {next_month_aqi}')

    # Append the forecast for the next month to the original time series
    ts_extended = ts.append(pd.Series(next_month_aqi, index=[ts.index[-1] + pd.offsets.MonthBegin(1)]))

    # Fit the ARIMA model to the extended time series
    model_ts_extended = ARIMA(ts_extended, order=pqd[index])
    model_ts_extended_fit = model_ts_extended.fit()

    # Use the trained model to make a forecast for the next 12 months
    forecast_next_year = model_ts_extended_fit.forecast(steps=12)

    # Create a date range for the next 12 months
    next_12_months = pd.date_range(start=ts_extended.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='M')

    # Convert the forecasts to a pandas Series with the date range as the index
    forecast_series = pd.Series(forecast_next_year[0], index=next_12_months)

    # Plot the original time series, the forecasts, and the confidence intervals
    plt.figure(figsize=(20,10))
    ts_extended.plot(label='Original')
    forecast_series.plot(label='Forecast')
    plt.fill_between(next_12_months, forecast_next_year[2][:, 0], forecast_next_year[2][:, 1], color='k', alpha=0.1)
    plt.legend()
    plt.show()

arima()