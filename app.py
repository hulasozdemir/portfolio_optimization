import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt

# Function to calculate portfolio return and volatility
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

# Function to calculate downside risk for Sortino Ratio
def calculate_downside_risk(returns, target_return=0):
    downside_returns = returns[returns < target_return]
    downside_risk = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    return downside_risk

# Function to calculate extended performance metrics including Sortino and Treynor Ratios
def calculate_extended_performance_metrics(portfolio_returns, market_returns, risk_free_rate=0):
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    downside_risk = calculate_downside_risk(portfolio_returns)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_risk
    
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    treynor_ratio = (annualized_return - risk_free_rate) / beta
    
    return annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, treynor_ratio

# Load stock data
def load_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    daily_returns = data.pct_change().dropna()
    return data, daily_returns

# Load market data (S&P 500)
market_data = yf.download('^GSPC', start='2018-07-12', end='2024-07-12')['Adj Close'].pct_change().dropna()
market_returns = market_data

# Function to calculate historical portfolio returns
def calculate_portfolio_returns(weights, returns):
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Portfolio Optimization Dashboard'),

    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=dt(2018, 7, 12),
        end_date=dt(2024, 7, 12),
        display_format='YYYY-MM-DD'
    ),

    dcc.Dropdown(
        id='stock-dropdown',
        options=[
        {'label': 'Apple Inc. (AAPL)', 'value': 'AAPL'},
        {'label': 'Microsoft Corp. (MSFT)', 'value': 'MSFT'},
        {'label': 'Alphabet Inc. (GOOGL)', 'value': 'GOOGL'},
        {'label': 'Amazon.com Inc. (AMZN)', 'value': 'AMZN'},
        {'label': 'Tesla Inc. (TSLA)', 'value': 'TSLA'},
        {'label': 'Meta Platforms Inc. (META)', 'value': 'META'},
        {'label': 'NVIDIA Corporation (NVDA)', 'value': 'NVDA'},
        {'label': 'Berkshire Hathaway Inc. (BRK-A)', 'value': 'BRK-A'},
        {'label': 'JPMorgan Chase & Co. (JPM)', 'value': 'JPM'},
        {'label': 'UnitedHealth Group Incorporated (UNH)', 'value': 'UNH'},
        {'label': 'Visa Inc. (V)', 'value': 'V'},
        {'label': 'Mastercard Inc. (MA)', 'value': 'MA'},
        {'label': 'The Walt Disney Company (DIS)', 'value': 'DIS'},
        {'label': 'Pfizer Inc. (PFE)', 'value': 'PFE'},
        {'label': 'The Coca-Cola Company (KO)', 'value': 'KO'},
        {'label': 'Cisco Systems Inc. (CSCO)', 'value': 'CSCO'},
        {'label': 'Adobe Inc. (ADBE)', 'value': 'ADBE'},
        {'label': 'Netflix Inc. (NFLX)', 'value': 'NFLX'},
        {'label': 'Intel Corporation (INTC)', 'value': 'INTC'},
        {'label': 'Walmart Inc. (WMT)', 'value': 'WMT'},
        {'label': 'Exxon Mobil Corporation (XOM)', 'value': 'XOM'},
        {'label': 'Chevron Corporation (CVX)', 'value': 'CVX'},
        {'label': 'Boeing Co. (BA)', 'value': 'BA'},
        {'label': 'IBM (International Business Machines Corp.) (IBM)', 'value': 'IBM'},
        {'label': 'AT&T Inc. (T)', 'value': 'T'},
        {'label': 'McDonald\'s Corp. (MCD)', 'value': 'MCD'},
        {'label': 'Nike Inc. (NKE)', 'value': 'NKE'},
        {'label': 'Texas Instruments Inc. (TXN)', 'value': 'TXN'},
        {'label': 'Goldman Sachs Group Inc. (GS)', 'value': 'GS'},
        {'label': 'Caterpillar Inc. (CAT)', 'value': 'CAT'},
        {'label': 'Home Depot Inc. (HD)', 'value': 'HD'},
        {'label': 'CVS Health Corporation (CVS)', 'value': 'CVS'},
        {'label': 'United Parcel Service Inc. (UPS)', 'value': 'UPS'},
        {'label': 'Lockheed Martin Corporation (LMT)', 'value': 'LMT'},
        {'label': 'Honda Motor Co., Ltd. (HMC)', 'value': 'HMC'},
        {'label': 'Oracle Corporation (ORCL)', 'value': 'ORCL'},
        {'label': 'Bristol-Myers Squibb Company (BMY)', 'value': 'BMY'},
        {'label': 'Salesforce.com Inc. (CRM)', 'value': 'CRM'},
        {'label': 'PayPal Holdings Inc. (PYPL)', 'value': 'PYPL'},
        {'label': 'Square Inc. (SQ)', 'value': 'SQ'},
        {'label': 'Starbucks Corporation (SBUX)', 'value': 'SBUX'},
        {'label': 'Uber Technologies Inc. (UBER)', 'value': 'UBER'},
        {'label': 'Alibaba Group Holding Ltd. (BABA)', 'value': 'BABA'},
        {'label': 'Tencent Holdings Ltd. (TCEHY)', 'value': 'TCEHY'},
        {'label': 'Baidu Inc. (BIDU)', 'value': 'BIDU'},
        {'label': 'NIO Inc. (NIO)', 'value': 'NIO'},
        {'label': 'JD.com Inc. (JD)', 'value': 'JD'},
        {'label': 'Moderna Inc. (MRNA)', 'value': 'MRNA'}
    ],
        value=['AAPL', 'MSFT'],  # Default selected values
        multi=True
    ),

    html.Label('Risk-Free Rate'),
    dcc.Slider(
        id='risk-free-rate-slider',
        min=0,
        max=0.1,
        step=0.01,
        value=0.01,
        marks={i / 100: f'{i}%' for i in range(0, 11)}
    ),

    html.Div([
        dcc.Graph(id='efficient-frontier'),
        dcc.Graph(id='backtest-comparison')
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Div(id='portfolio-weights')
])

# Define callback to update the graphs and weights based on user input
@app.callback(
    [Output('efficient-frontier', 'figure'),
     Output('backtest-comparison', 'figure'),
     Output('portfolio-weights', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('risk-free-rate-slider', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_stocks, risk_free_rate, start_date, end_date):
    # Load stock data
    tickers = selected_stocks
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    stock_data, daily_returns = load_stock_data(tickers, start_date, end_date)

    # Check for empty data
    if daily_returns.empty:
        return {}, {}, "No data available for the selected date range."

    # Generate random portfolios
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(stock_data.columns))
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)
        try:
            portfolio_return, portfolio_volatility = portfolio_performance(weights, daily_returns)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio
        except (ZeroDivisionError, ValueError) as e:
            print(f"Error in portfolio calculation: {e}")
            continue

    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    # Debugging print statements
    # print(results_df.head())
    # print(results_df.describe())
    
    # Check for NaN values
    if results_df['Sharpe Ratio'].isna().all():
        return {}, {}, "No valid Sharpe Ratio found."

    # Identify the portfolio with the maximum Sharpe ratio
    try:
        max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
        max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
        max_sharpe_weights = weights_record[max_sharpe_idx]
    except ValueError as e:
        print(f"Error finding max Sharpe index: {e}")
        return {}, {}, "Error in Sharpe Ratio calculation."

    # Identify the portfolio with the minimum volatility
    try:
        min_vol_idx = results_df['Volatility'].idxmin()
        min_vol_portfolio = results_df.loc[min_vol_idx]
        min_vol_weights = weights_record[min_vol_idx]
    except ValueError as e:
        print(f"Error finding min volatility index: {e}")
        return {}, {}, "Error in volatility calculation."

    # Calculate cumulative returns for the portfolios
    min_vol_returns = calculate_portfolio_returns(min_vol_weights, daily_returns)
    max_sharpe_returns = calculate_portfolio_returns(max_sharpe_weights, daily_returns)
    
    cumulative_min_vol_returns = (1 + min_vol_returns).cumprod() * 1000 # Start with an initial investment of $1000
    cumulative_max_sharpe_returns = (1 + max_sharpe_returns).cumprod() * 1000
    
    # Create the comparison graph
    backtest_comparison = go.Figure()
    backtest_comparison.add_trace(go.Scatter(
        x=cumulative_min_vol_returns.index,
        y=cumulative_min_vol_returns,
        mode='lines',
        name='Min Volatility Portfolio'
    ))
    backtest_comparison.add_trace(go.Scatter(
        x=cumulative_max_sharpe_returns.index,
        y=cumulative_max_sharpe_returns,
        mode='lines',
        name='Max Sharpe Ratio Portfolio'
    ))
    backtest_comparison.update_layout(
        title='Backtest Comparison of Portfolio Values',
        xaxis_title='Date',
        yaxis_title='Cumulative Return'
    )

    # Create efficient frontier graph
    fig_efficient_frontier = go.Figure()
    fig_efficient_frontier.add_trace(go.Scatter(
        x=results_df['Volatility'],
        y=results_df['Return'],
        mode='markers',
        marker=dict(color=results_df['Sharpe Ratio'], colorscale='Viridis', showscale=True),
        text=[f'Sharpe: {sharpe:.2f}' for sharpe in results_df['Sharpe Ratio']],
        showlegend=False
    ))

    # Add markers for Min Volatility and Max Sharpe Ratio portfolios
    fig_efficient_frontier.add_trace(go.Scatter(
        x=[min_vol_portfolio['Volatility']],
        y=[min_vol_portfolio['Return']],
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Min Volatility Portfolio',
        showlegend=False
    ))
    
    fig_efficient_frontier.add_trace(go.Scatter(
        x=[max_sharpe_portfolio['Volatility']],
        y=[max_sharpe_portfolio['Return']],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name='Max Sharpe Ratio Portfolio',
        showlegend=False
    ))

    fig_efficient_frontier.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Return'
    )

    # Create table data
    table_data = {
        'Stock': tickers,
        'Min Volatility': min_vol_weights,
        'Max Sharpe Ratio': max_sharpe_weights
    }
    
    # Convert to DataFrame for display
    table_df = pd.DataFrame(table_data)
    
    # Create the table component
    table = dash_table.DataTable(
        id='portfolio-table',
        columns=[{'name': i, 'id': i} for i in table_df.columns],
        data=table_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    return fig_efficient_frontier, backtest_comparison, table

# Run the app. Use the port 8050
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
