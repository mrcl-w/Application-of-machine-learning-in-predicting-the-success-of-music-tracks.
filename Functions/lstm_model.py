import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_and_evaluate_model(features: list, data: pd.DataFrame, SEQ_LEN: int = 4) -> float:
    #prepare sequences
    #prepare data
    groups = data.groupby(['Song', 'Artist'])

    X = []
    y = []
    target_dates = []
    meta = []

    for (song, artist), group in groups:
        group = group.sort_values('Date')

        data = group[features].values
        ranks = group['Rank'].values
        dates = group['Date'].values

        for i in range(len(group) - SEQ_LEN):
            X_seq = data[i:i+SEQ_LEN]
            y_target = ranks[i+SEQ_LEN]
            target_date = dates[i+SEQ_LEN]

            X.append(X_seq)
            y.append(y_target)
            target_dates.append(target_date)

            meta.append({
                'Song': song,
                'Artist': artist,
                'Date': target_date,
                'target_next_rank': y_target
            })

    X = np.array(X)
    y = np.array(y)
    target_dates = np.array(target_dates)

    meta = pd.DataFrame(meta)

    #split into train and test based on date
    target_dates = pd.to_datetime(target_dates)
    threshold_date = pd.to_datetime("2023-01-01")

    train_mask = target_dates < threshold_date
    test_mask = target_dates >= threshold_date

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    #scale features
    scaler = MinMaxScaler()
    samples_train, timesteps, features_count = X_train.shape
    samples_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, features_count)
    X_test_2d = X_test.reshape(-1, features_count)

    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    X_train = X_train_scaled.reshape(samples_train, timesteps, features_count)
    X_test = X_test_scaled.reshape(samples_test, timesteps, features_count)

    #scale target if needed
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    #build model
    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, features_count)),
        Dense(32, activation='relu'),
        Dense(1)
        ])
    
    #compile model
    model.compile(
        optimizer='adam',
        loss=Huber(delta=1.0),
        metrics=['mae']
    )

    #train model
    history = model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_test, y_test_scaled),
        epochs=20,
        batch_size=32
     )
    #evaluate model
    loss, mae = model.evaluate(X_test, y_test_scaled)
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_test.reshape(-1, 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("MAE:", mae)
    print("RMSE:", rmse)
    return mae


