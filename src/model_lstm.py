"""
Modul Pembangunan dan Pelatihan Model LSTM
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
"""

import numpy as np


def build_lstm_model(input_shape: tuple,
                     lstm_units: int = 32,
                     dropout_rate: float = 0.1,
                     dense_units: int = 16,
                     learning_rate: float = 0.001):
    """
    Membangun arsitektur Sequential LSTM untuk peramalan time series regresi.
    input_shape: (sequence_length, n_features)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=False, name='lstm_layer_1'),
        Dropout(rate=dropout_rate, name='dropout_1'),
        Dense(units=dense_units, activation='relu', name='dense_hidden'),
        Dense(units=1, activation='linear', name='output_power')
    ], name=f'LSTM_Load_Forecaster_U{lstm_units}')

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model


def train_lstm_model(model,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     epochs: int = 20,
                     batch_size: int = 32,
                     patience: int = 5,
                     checkpoint_path: str = None):
    """
    Melatih model LSTM dengan Early Stopping dan Model Checkpoint.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history
