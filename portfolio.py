import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

# Streamlit app configuration
st.title("Portfolio Analysis")

# List of available tickers
available_tickers = {
    '^GSPC': 'S&P 500',
    '^STOXX50E': 'EURO STOXX 50',
    '^FTSE': 'FTSE 100',
}

# Function to get user-selected tickers
def get_selected_tickers():
    return st.multiselect("Select two indices for the portfolio:", list(available_tickers.keys()), default=list(available_tickers.keys())[:2])

selected_tickers = get_selected_tickers()

# Check if the correct number of tickers are selected
if len(selected_tickers) != 2:
    st.error("Please select exactly 2 indices.")
else:
    # Function to get weights from user
    def get_weights():
        weights = []
        for ticker in selected_tickers:
            weight = st.number_input(f"Enter weight for {available_tickers[ticker]} ({ticker}):", min_value=0.0, max_value=1.0, step=0.01)
            weights.append(weight)
        if sum(weights) == 1:
            return np.array(weights)
        else:
            st.error("The weights must sum up to 1. Please adjust the values.")
            return None

    weights = None
    while weights is None:
        weights = get_weights()

    # Function to get choice of period from user
    period_options = ["1y", "3y", "5y", "10y"]
    selected_period = st.selectbox("Select the historical data period:", period_options, index=0)

    # Download historical prices for the selected period
    data = yf.download(selected_tickers, period=selected_period)['Adj Close']

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)

    # Calculate expected portfolio return and portfolio standard deviation
    expected_portfolio_return = np.mean(portfolio_returns)
    portfolio_std_dev = np.std(portfolio_returns)

    # Assuming a risk-free rate of 2%
    risk_free_rate = 0.02 / 252  # Daily risk-free rate

    # Calculate Sharpe ratio
    sharpe_ratio = (expected_portfolio_return - risk_free_rate) / portfolio_std_dev

    # Display the results
    expected_portfolio_return_annual = expected_portfolio_return * 252
    portfolio_std_dev_annual = portfolio_std_dev * np.sqrt(252)

    st.write(f"Expected Portfolio Return: {expected_portfolio_return_annual:.2%}")
    st.write(f"Portfolio Standard Deviation: {portfolio_std_dev_annual:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # VaR Calculation
    confidence_level = 0.95
    VaR = norm.ppf(1 - confidence_level) * portfolio_std_dev - expected_portfolio_return
    VaR_annual = VaR * np.sqrt(252)

    st.write(f"Value at Risk (VaR) (Daily): {VaR:.2%}")
    st.write(f"Value at Risk (VaR) (Annual): {VaR_annual:.2%}")

    # Plotting Portfolio Allocation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(weights, labels=[available_tickers[ticker] for ticker in selected_tickers], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
    ax.set_title('Portfolio Allocation')
    st.pyplot(fig)

