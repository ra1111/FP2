import streamlit as st
import cloudpickle
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import streamlit as st

# Function Definitions (shortened for brevity)

def get_stock_symbols():
    stocks = st.text_input("Enter 10 stock symbols separated by commas:").split(',')
    if len(stocks) != 10:
        st.error("Please enter exactly 10 stock symbols.")
        st.stop()
    stock_symbols = [stock.strip().upper() + ".NS" for stock in stocks]
    return stock_symbols

def get_date_input(prompt):
    date_str = st.text_input(prompt)
    try:
        pd.to_datetime(date_str)
        return date_str
    except ValueError:
        st.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")
        st.stop()

def fetch_data(stock_symbols, start_date, end_date):
    data = {}
    for symbol in stock_symbols:
        ticker = yf.Ticker(symbol)
        data[symbol] = ticker.history(start=start_date, end=end_date)
    return data

def preprocess_data(data):
    processed_data = {}
    for symbol, df in data.items():
        df = df[['Close']].copy()
        df.dropna(inplace=True)
        processed_data[symbol] = df
    return processed_data

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(data, symbol):
    df = data[symbol]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    train_data_len = int(np.ceil(len(scaled_data) * .95))
    train_data = scaled_data[0:int(train_data_len), :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    return model, scaler

def predict_lstm(data, model, scaler, symbol, days=30):
    df = data[symbol]
    scaled_data = scaler.transform(df)

    test_data = scaled_data[-60:, :]
    x_test = [test_data]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = []
    for _ in range(days):
        predicted_price = model.predict(x_test)
        predicted_prices.append(predicted_price[0][0])
        predicted_price_reshaped = np.reshape(predicted_price[0][0], (1, 1, 1))
        x_test = np.append(x_test[:, 1:, :], predicted_price_reshaped, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    
    return predicted_prices

def get_short_term_predictions(stock_symbols, processed_data):
    short_term_predictions = {}
    
    for symbol in stock_symbols:
        model, scaler = train_lstm(processed_data, symbol)
        predictions = predict_lstm(processed_data, model, scaler, symbol)
        short_term_predictions[symbol] = predictions
        
    return short_term_predictions

def plot_predictions(stock_data, predictions, stock_symbols):
    for symbol in stock_symbols:
        plt.figure(figsize=(12, 6))
        actual_data = stock_data[symbol]['Close']
        prediction_data = predictions[symbol]
        prediction_dates = pd.date_range(start=actual_data.index[-1], periods=len(prediction_data)+1, closed='right')

        plt.plot(actual_data.index, actual_data, label='Actual Prices', color='blue')
        plt.plot(prediction_dates, prediction_data, label='Predicted Prices', color='red')
        plt.title(f"Stock Price Prediction for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

def train_arima(processed_data, stock_symbols):
    df = processed_data[stock_symbols]
    model = ARIMA(df, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def predict_arima(model_fit, days=120):
    forecast = model_fit.forecast(steps=days)
    return forecast.values

def get_long_term_predictions(stock_symbols, processed_data):
    long_term_predictions = {}
    
    for symbol in stock_symbols:
        model_fit = train_arima(processed_data, symbol)
        predictions = predict_arima(model_fit)
        long_term_predictions[symbol] = predictions
        
    return long_term_predictions

def plot_long_term_predictions(stock_data, predictions, stock_symbols):
    for symbol in stock_symbols:
        plt.figure(figsize=(12, 6))
        actual_data = stock_data[symbol]['Close']
        prediction_data = predictions[symbol]
        prediction_dates = pd.date_range(start=actual_data.index[-1], periods=len(prediction_data)+1, closed='right')

        plt.plot(actual_data.index, actual_data, label='Actual Prices', color='blue')
        plt.plot(prediction_dates, prediction_data, label='Predicted Prices', color='green')
        plt.title(f"Long-term Stock Price Prediction for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return volatility, returns, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def portfolio_optimization(predictions, processed_data):
    short_term_prices = pd.DataFrame({symbol: predictions[symbol].flatten() for symbol in predictions})
    long_term_prices = pd.DataFrame({symbol: predictions[symbol] for symbol in predictions})

    # Short-term optimization
    short_term_mu = short_term_prices.pct_change().mean()
    short_term_S = short_term_prices.pct_change().cov()
    short_term_weights = optimize_portfolio(short_term_mu, short_term_S)

    # Long-term optimization
    long_term_mu = long_term_prices.pct_change().mean()
    long_term_S = long_term_prices.pct_change().cov()
    long_term_weights = optimize_portfolio(long_term_mu, long_term_S)

    return short_term_weights, long_term_weights, short_term_prices, long_term_prices

def get_and_print_optimized_portfolios(long_term_predictions, processed_data):
    short_term_portfolio, long_term_portfolio, short_term_prices, long_term_prices = portfolio_optimization(long_term_predictions, processed_data)
    return short_term_portfolio, long_term_portfolio, short_term_prices, long_term_prices

def plot_portfolio_performance(weights, mean_returns, cov_matrix, title):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (returns - 0.01) / volatility

    plt.scatter(volatility, returns, c=sharpe_ratio, marker='o')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title(title)
    plt.colorbar(label='Sharpe Ratio')
    st.pyplot(plt)

def calculate_and_plot_portfolio_performance(short_term_portfolio, short_term_prices, long_term_portfolio, long_term_prices):
    # Calculate performance for short-term portfolio
    short_term_mu = short_term_prices.pct_change().mean()
    short_term_S = short_term_prices.pct_change().cov()
    plot_portfolio_performance(short_term_portfolio, short_term_mu, short_term_S, "Short-term Portfolio Performance")

    # Calculate performance for long-term portfolio
    long_term_mu = long_term_prices.pct_change().mean()
    long_term_S = long_term_prices.pct_change().cov()
    plot_portfolio_performance(long_term_portfolio, long_term_mu, long_term_S, "Long-term Portfolio Performance")
    return short_term_mu, short_term_S, long_term_mu, long_term_S

def plot_performance(predictions, title):
    for symbol, pred in predictions.items():
        plt.plot(pred, label=symbol)
        plt.title(title)
        plt.xlabel("Days")
        plt.ylabel("Predicted Price")
        plt.legend()
        st.pyplot(plt)



def main():
    st.title("Stock Value Prediction and Portfolio Optimization - Group-17 AMPBA Co'24 Summer")
    # HTML styling for the app
html_temp = """
<div style="background-color: tomato; padding: 10px;">
<h2 style="color: white; text-align: center;">Streamlit Stock Prediction App FP-2 </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# User input for stock tickers
stock_symbols = get_stock_symbols()

# User input for desired timeframe
start_date = st.date_input("Select start date", pd.to_datetime("2012-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("2022-01-01"))

if st.button("Fetch Data"):
    # Fetching stock data
    stock_data = fetch_data(stock_symbols, start_date, end_date)
    processed_data = preprocess_data(stock_data)

    # Short-term predictions
    st.subheader("Short-term Predictions")
    short_term_predictions = get_short_term_predictions(stock_symbols, processed_data)
    plot_predictions(stock_data, short_term_predictions, stock_symbols)

    # Long-term predictions
    st.subheader("Long-term Predictions")
    long_term_predictions = get_long_term_predictions(stock_symbols, processed_data)
    plot_long_term_predictions(stock_data, long_term_predictions, stock_symbols)

    # Portfolio optimization
    st.subheader("Portfolio Optimization")
    short_term_portfolio, long_term_portfolio, short_term_prices, long_term_prices = get_and_print_optimized_portfolios(long_term_predictions, processed_data)
    st.write("Short-term Portfolio Weights:", short_term_portfolio)
    st.write("Long-term Portfolio Weights:", long_term_portfolio)

    # Portfolio performance
    st.subheader("Portfolio Performance")
    short_term_mu, short_term_S, long_term_mu, long_term_S = calculate_and_plot_portfolio_performance(short_term_portfolio, short_term_prices, long_term_portfolio, long_term_prices)


if __name__ == '__main__':
    main()

