# # ---------------------------------------- LSTM ----------------------------------------
import pandas as pd
import numpy as np


def test_func():
    lstm_df = pd.read_excel("./content/cancer patient data sets.xlsx")

    lstm_df.head()

    lstm_df = lstm_df.dropna()

    lstm_df.info()

    import math

    dataset = lstm_df.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    training_data_len

    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_data = sc.fit_transform(lstm_df[["Age", "AQI"]])
    scaled_data

    ################3
    scaler = MinMaxScaler()
    lstm_df[["Age", "AQI"]] = scaler.fit_transform(lstm_df[["Age", "AQI"]])

    from sklearn.model_selection import train_test_split

    X = lstm_df[
        [
            "Age",
            "Gender",
            "AQI",
            "Dust Allergy",
            "OccuPational Hazards",
            "Genetic Risk",
            "chronic Lung Disease",
            "Smoking",
            "Passive Smoker",
            "Clubbing of Finger Nails",
            "Frequent Cold",
        ]
    ]
    y = lstm_df["Level"]

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    num = 60
    for i in range(num, len(train_data)):
        x_train.append(train_data[i - num : i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    # model = Sequential([
    #  LSTM(units=64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    #   Dense(1, activation='sigmoid')  # Assuming binary classification
    # ])
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import InputLayer

    model_lstm = Sequential()

    model_lstm.add(InputLayer((12, 1)))

    model_lstm.add(LSTM(50))

    model_lstm.add(Dense(34, "relu"))
    # model_lstm.add(Dropout(0.25))

    model_lstm.add(Dense(15, "relu"))

    model_lstm.add(Dense(1, "relu"))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    model.fit(x_train, y_train, batch_size=1, epochs=1)

    model_lstm.summary()

    test_data = scaled_data[training_data_len - 60 :, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(num, len(test_data)):
        x_test.append(test_data[i - num : i, 0])

    x_test = np.array(x_test)
    print(x_test.shape)

    print(y_test.shape)

    y_test = np.random.rand(100, 2)
    y_test = np.concatenate((y_test, np.zeros((100, 2))), axis=0)

    # print(x_test)

    # print(y_test)

    print("Now we do za predict")

    # predictions = model.predict(x_test)
    # predictions = predictions.reshape(-1,2)
    # predictions = sc.inverse_transform(predictions)
    # predictions= predictions.squeeze()

    # Assuming x_test has shape (batch_size, 6, 10)
    # x_test = x_test.reshape(200, 11, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 60, 1)

    predictions = model.predict(x_test_reshaped)
    predictions = predictions.reshape(-1, 2)
    predictions = sc.inverse_transform(predictions)
    print("y u stop?")
    print(predictions)

    low_threshold = 1.10  # Adjust this value based on your specific problem
    high_threshold = 1.14  # Adjust this value based on your specific problem

    thatarray = []

    # Create an empty list to store the categories
    categories = []

    # Categorize the predictions
    for prediction in predictions:
        per_cent = prediction[0] / (prediction[0] + prediction[1])
        thatarray.append(per_cent)
    x = thatarray
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    print(x_norm)

    print("model is za done :)")

    # make one row as df with col Age, Gender, AQI, Dust Allergy, OccuPational Hazards, Genetic Risk, chronic Lung Disease, Smoking, Passive Smoker, Clubbing of Finger Nails, Frequent Cold

    single_row_df = {"Age":21, "Gender": 1, "AQI": 4, "Dust Allergy": 2, "OccuPational Hazards": 1, "Genetic Risk": 2, "chronic Lung Disease": 5, "Smoking": 1, "Passive Smoker": 2, "Clubbing of Finger Nails": 4, "Frequent Cold": 6}
    single_row_df = pd.DataFrame(single_row_df, index=[0])

    # Scale the 'Age' and 'AQI' columns
    single_row_df[["Age", "AQI"]] = scaler.transform(single_row_df[["Age", "AQI"]])

    # Select the necessary columns
    X_single = single_row_df[
        [
            "Age",
            "Gender",
            "AQI",
            "Dust Allergy",
            "OccuPational Hazards",
            "Genetic Risk",
            "chronic Lung Disease",
            "Smoking",
            "Passive Smoker",
            "Clubbing of Finger Nails",
            "Frequent Cold",
        ]
    ]

    # Convert the DataFrame to a numpy array and reshape it
    X_single = np.array(X_single)
    X_single = X_single.reshape(1, X_single.shape[0], 1)

    # Use the model to make a prediction
    single_prediction = model.predict(X_single)
    return [x_norm, "issa done"]


test_func()
