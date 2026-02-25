import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def train_autoencoder(model, X_train_normal):
    history = model.fit(
        X_train_normal,
        X_train_normal,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        shuffle=True
    )
    return history

def predict_autoencoder(model, X_test, X_train_normal):
    train_recon = model.predict(X_train_normal)
    train_error = np.mean(np.square(X_train_normal - train_recon), axis=1)
    threshold = np.percentile(train_error, 95)

    test_recon = model.predict(X_test)
    test_error = np.mean(np.square(X_test - test_recon), axis=1)

    predictions = (test_error > threshold).astype(int)

    return predictions, test_error
