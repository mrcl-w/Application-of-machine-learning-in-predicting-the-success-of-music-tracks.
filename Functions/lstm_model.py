import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    tf.config.experimental.enable_op_determinism()


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
    # Build the LSTM model architecture
    
    # Create a Sequential model
    model = Sequential([
        Input(shape=(seq_len, features_count))
    ])

    # Add LSTM layers with specified units and dropout
    for i, units in enumerate(lstm_units):
        model.add(
            LSTM(
                units,
                return_sequences=i < len(lstm_units) - 1
            )
        )

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Add Dense layers with specified units and dropout
    for units in dense_units:
        model.add(Dense(units, activation="relu"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Add the output layer with a single neuron for regression
    model.add(Dense(1))

    # Compile the model with Adam optimizer, Huber loss, and mean absolute error metric
    model.compile(
        optimizer="adam",
        loss=Huber(),
        metrics=["mae"]
    )

    return model

def build_and_evaluate_model(
    prepared_data,
    seq_len,
    lstm_units,
    dense_units,
    dropout_rate,
    seed,
    epochs=50,
    batch_size=32
):
    set_seed(seed)


    model = build_lstm_model(
        seq_len=seq_len,
        features_count=prepared_data["features_count"],
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        prepared_data["X_train"],
        prepared_data["y_train_scaled"],
        validation_data=(
            prepared_data["X_val"],
            prepared_data["y_val_scaled"]
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping],
        verbose=0
    )

    pred_scaled = model.predict(
        prepared_data["X_test"],
        verbose=0
    )

    pred = prepared_data["y_scaler"].inverse_transform(pred_scaled).reshape(-1)
    y_true = prepared_data["y_test"]

    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))

    return model, history, mae, rmse
