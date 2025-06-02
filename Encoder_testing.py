import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path, scaler=None, window_size=10, fit_scaler=False):
    df = pd.read_csv(file_path)
    features = ['hold_time', 'latency']
    
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[features])
    else:
        if fit_scaler:
            scaler.fit(df[features])
        data_scaled = scaler.transform(df[features])
    
    sequences = []
    for i in range(len(data_scaled) - window_size + 1):
        sequences.append(data_scaled[i:i+window_size])
    sequences = np.array(sequences)
    return sequences, scaler

def build_lstm_autoencoder(window_size, feature_dim, latent_dim=16):
    from tensorflow.keras.layers import Dropout
    inputs = Input(shape=(window_size, feature_dim))
    encoded = LSTM(latent_dim, activation='relu', dropout=0.2, recurrent_dropout=0.2)(inputs)
    decoded = RepeatVector(window_size)(encoded)
    decoded = LSTM(feature_dim, activation='linear', return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_model(autoencoder, X_train, X_val, epochs=50, batch_size=32):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return history

def calculate_mse(autoencoder, X):
    X_pred = autoencoder.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
    return mse

def find_threshold(mse_train, percentile=95):
    return np.percentile(mse_train, percentile)

def calculate_legitimacy_probability(mse, alpha=90):
    prob = np.exp(-alpha * (mse / np.max(mse)))
    return prob

if __name__ == "__main__":
    WINDOW_SIZE = 10
    LEARN_FILE = 'learn_ds.csv'
    TEST_FILE = 'test_ds_kate.csv'

    data, scaler = load_and_preprocess(LEARN_FILE, window_size=WINDOW_SIZE)
    X_train, X_val = train_test_split(data, test_size=0.1, random_state=42)

    feature_dim = data.shape[2]
    autoencoder = build_lstm_autoencoder(WINDOW_SIZE, feature_dim)

    train_model(autoencoder, X_train, X_val)

    mse_train = calculate_mse(autoencoder, X_train)
    threshold = find_threshold(mse_train, percentile=95)
    print(f"Порог аномалии (95-й перцентиль): {threshold:.5f}")

    test_data, _ = load_and_preprocess(TEST_FILE, scaler=scaler, window_size=WINDOW_SIZE)
    mse_test = calculate_mse(autoencoder, test_data)

    probabilities = calculate_legitimacy_probability(mse_test)
    avg_prob = np.mean(probabilities)
    print(f"Средняя вероятность легитимности пользователя: {avg_prob:.4f}")
