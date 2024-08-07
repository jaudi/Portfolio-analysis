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
    '^N225': 'Nikkei 225',
    '^HSI': 'Hang Seng Index',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'NASDAQ Composite',
    '^RUT': 'Russell 2000',
    '^VIX': 'CBOE Volatility Index'
}

# Function to get user-selected tickers
def get_selected_tickers():
    return st.multiselect("Select up to four indices for your portfolio:", list(available_tickers.keys()), default=list(available_tickers.keys())[:4])

selected_tickers = get_selected_tickers()

# Check if the correct number of tickers are selected
if not (1 <= len(selected_tickers) <= 4):
    st.error("Please select between one and four indices.")
else:
    # Initialize default weights (you can adjust these defaults as needed)
    default_weights = [0.25] * len(selected_tickers)

    # Function to get weights from user
    def get_weights():
        weights = []
        for i, ticker in enumerate(selected_tickers):
            weight = st.number_input(f"Enter weight for {available_tickers[ticker]} ({ticker}):", min_value=0.0, max_value=1.0, step=0.01, value=default_weights[i])
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
    def get_choice():
        options = ["1y", "2y", '5y', '10y', 'ytd']
        choice = st.selectbox("Choose the time period:", options)
        return choice

    selected_period = get_choice()

    # Download historical prices for the selected period
    try:
        data = yf.download(selected_tickers, period=selected_period, progress=False)['Adj Close']
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        data = None

    if data is not None:
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
        import plotly.graph_objects as go

        # Create figure
        fig = go.Figure()
        
        # Add trace for pie chart
        fig.add_trace(go.Pie(labels=[available_tickers[ticker] for ticker in selected_tickers], values=weights, hole=0.4, textinfo='label+percent'))
        
        fig.update_layout(title_text="Portfolio Allocation", title_x=0.5)
        
        st.plotly_chart(fig)

        
        # Explanation of VaR and Sharpe Ratio
        st.markdown("---")
        st.subheader("Definitions")
        st.markdown("### Value at Risk (VaR)")
        st.markdown("Value at Risk (VaR) is a statistical measure used to quantify the level of financial risk within a portfolio over a specified time frame. It estimates the maximum potential loss in value of a portfolio or an investment, under normal market conditions, at a certain confidence level (typically 95% or 99%).")
        st.markdown("### Sharpe Ratio")
        st.markdown("Sharpe Ratio is a measure of risk-adjusted return, developed by Nobel laureate William F. Sharpe. It evaluates the performance of an investment (or a portfolio) compared to a risk-free asset, after adjusting for its risk. A higher Sharpe Ratio indicates better risk-adjusted performance.")

        # Disclaimer
        st.markdown("---")
        st.subheader("Disclaimer")
        st.markdown("""
        The information presented in this app is for educational and informational purposes only. It is not intended as financial advice or as a recommendation to buy or sell any securities. Investing involves risks, including the potential loss of principal. Past performance does not guarantee future results. Users should conduct their own research and consult with a qualified financial advisor before making investment decisions.
        """)

    
