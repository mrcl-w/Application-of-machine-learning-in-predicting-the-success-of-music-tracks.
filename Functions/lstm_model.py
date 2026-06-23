import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_lstm_data(
    dataframe,
    features,
    seq_len=4,
    val_date="2012-01-01",
    test_date="2014-01-01"
):
    groups = dataframe.groupby(['Song', 'Artist'])

    # Prepare the input sequences (X) and target values (y) for the LSTM model
    X, y, target_dates = [], [], []

    for (_, _), group in groups:
        group = group.sort_values('Date')

        feature_data = group[features].values
        ranks = group['Rank'].values
        dates = group['Date'].values

        for i in range(len(group) - seq_len):
            X.append(feature_data[i:i + seq_len])
            y.append(ranks[i + seq_len])
            target_dates.append(dates[i + seq_len])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)


    target_dates = pd.to_datetime(target_dates)

    train_mask = target_dates < val_date
    val_mask = (target_dates >= val_date) & (target_dates < test_date)
    test_mask = target_dates >= test_date

    # Split the data into training and testing sets based on the threshold date
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    # Scale the features and target values using MinMaxScaler
    samples_train, timesteps, features_count = X_train.shape
    samples_test = X_test.shape[0]

    x_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(
        X_train.reshape(-1, features_count)
    ).reshape(samples_train, timesteps, features_count)

    X_test = x_scaler.transform(
        X_test.reshape(-1, features_count)
    ).reshape(samples_test, timesteps, features_count)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "features_count": features_count
    }

def build_lstm_model(
    seq_len,
    features_count,
    lstm_units,
    dense_units,
    dropout_rate
):
    model = Sequential([
        Input(shape=(seq_len, features_count))
    ])

    for i, units in enumerate(lstm_units):
        model.add(
            LSTM(
                units,
                return_sequences=i < len(lstm_units) - 1
            )
        )

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    for units in dense_units:
        model.add(Dense(units, activation="relu"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=Huber(),
        metrics=["mae"]
    )

    return model















def build_and_evaluate_model(
    features: list,
    dataframe: pd.DataFrame,
    seq_len: int = 4,
    lstm_units: list[int] = [64],
    dense_units: list[int] = [32],
    dropout_rate: float = 0.0,
    epochs: int = 20,
    batch_size: int = 32
):
    # Group the dataframe by 'Song' and 'Artist' to create sequences for each song-artist pair
    groups = dataframe.groupby(['Song', 'Artist'])

    # Prepare the input sequences (X) and target values (y) for the LSTM model
    X, y, target_dates = [], [], []

    for (_, _), group in groups:
        group = group.sort_values('Date')

        feature_data = group[features].values
        ranks = group['Rank'].values
        dates = group['Date'].values

        for i in range(len(group) - seq_len):
            X.append(feature_data[i:i + seq_len])
            y.append(ranks[i + seq_len])
            target_dates.append(dates[i + seq_len])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    target_dates = pd.to_datetime(target_dates)

    threshold_date = pd.Timestamp("2014-01-01")

    train_mask = target_dates < threshold_date
    test_mask = target_dates >= threshold_date

    # Split the data into training and testing sets based on the threshold date
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Scale the features and target values using MinMaxScaler
    samples_train, timesteps, features_count = X_train.shape
    samples_test = X_test.shape[0]

    x_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(
        X_train.reshape(-1, features_count)
    ).reshape(samples_train, timesteps, features_count)

    X_test = x_scaler.transform(
        X_test.reshape(-1, features_count)
    ).reshape(samples_test, timesteps, features_count)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(seq_len, features_count)))
    
    # Add LSTM layers with specified units and dropout
    for index, units in enumerate(lstm_units):
        is_last_lstm = index == len(lstm_units) - 1

        model.add(
            LSTM(
                units,
                return_sequences=not is_last_lstm
            )
        )

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    for units in dense_units:
        model.add(Dense(units, activation='relu'))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss=Huber(delta=1.0),
        metrics=['mae']
    )

    # Define callbacks for learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
)

    history = model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_test, y_test_scaled),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr, early_stopping]
    )

    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test, y_pred.reshape(-1))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred.reshape(-1)))

    print("MAE:", mae)
    print("RMSE:", rmse)

    return model, history, mae, rmse, x_scaler, y_scaler

