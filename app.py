import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Price Trend Prediction with LSTM')

# Load trained model
def load_lstm_model(path):
    return load_model(path)

model = load_lstm_model('lstm_stock_model.h5')

# Helper: create sequences
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

# Upload CSV
data_file = st.file_uploader('Upload a CSV file with stock prices (must have a Close column)', type=['csv'])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write('Data Preview:', df.head())

    if 'Close' not in df.columns:
        st.error('CSV must contain a "Close" column.')
    else:
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Ensure 'Close' column is numeric and drop NaNs
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        close_prices = df['Close'].dropna().values.reshape(-1, 1)

        # Calculate Moving Average (MA)
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        # Calculate RSI
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        df['RSI_14'] = compute_rsi(df['Close'])

        # Show indicators in UI
        st.write('With Technical Indicators:', df[['Close', 'MA_20', 'RSI_14']].tail(30))

        if len(close_prices) == 0:
            st.error('No valid numeric values found in the Close column.')
        else:
            scaled_data = scaler.fit_transform(close_prices)
            # Continue with the rest of your code...


        # Sequence length (should match training)
        SEQ_LENGTH = 60
        if len(scaled_data) <= SEQ_LENGTH:
            st.error(f'Not enough data for sequence length {SEQ_LENGTH}.')
        else:
            X_input = create_sequences(scaled_data, SEQ_LENGTH)
            X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))

            # Predict
            y_pred = model.predict(X_input)
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_true_inv = scaler.inverse_transform(scaled_data[SEQ_LENGTH:])

            # Plot with MA_20 and RSI_14
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            # Main price plot
            ax1.plot(y_true_inv, label='Actual Price')
            ax1.plot(y_pred_inv, label='Predicted Price')
            # Plot MA_20 (last N points to match y_true_inv length)
            ma_20 = df['MA_20'].values[-len(y_true_inv):]
            ax1.plot(ma_20, label='MA 20', linestyle='--')
            ax1.set_title('Actual vs. Predicted Stock Price with MA 20')
            ax1.set_ylabel('Stock Price')
            ax1.legend()
            # RSI subplot
            rsi_14 = df['RSI_14'].values[-len(y_true_inv):]
            ax2.plot(rsi_14, label='RSI 14', color='purple')
            ax2.axhline(70, color='red', linestyle='--', linewidth=1)
            ax2.axhline(30, color='green', linestyle='--', linewidth=1)
            ax2.set_title('RSI 14')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('RSI')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # Metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            st.write(f'**RMSE:** {rmse:.2f}')
            st.write(f'**MAE:** {mae:.2f}')

st.markdown('---')
st.markdown('Developed with Streamlit and Keras')
