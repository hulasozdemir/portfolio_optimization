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
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# stock list for the dropdown
stock_options = [
    {'label': 'Apple Inc. (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft Corp. (MSFT)', 'value': 'MSFT'},
    {'label': 'Alphabet Inc. (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Amazon.com Inc. (AMZN)', 'value': 'AMZN'},
    {'label': 'Tesla Inc. (TSLA)', 'value': 'TSLA'},
    {'label': 'Meta Platforms Inc. (META)', 'value': 'META'},
    {'label': 'NVIDIA Corporation (NVDA)', 'value': 'NVDA'},
    {'label': 'Berkshire Hathaway Inc. (BRK.A)', 'value': 'BRK.A'},
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
]

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Portfolio Optimization Dashboard"), width=12)
    ], justify='center'),

    dbc.Row([
        dbc.Col(dcc.DatePickerRange(
            id='date-picker-range',
            start_date=dt(2018, 7, 12),
            end_date=dt(2024, 7, 12),
            display_format='YYYY-MM-DD'
        ), width=6)
    ], justify='center'),

    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='stock-dropdown',
            options=stock_options,
            value=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            multi=True
        ), width=6)
    ], justify='center'),

    dbc.Row([
        dbc.Col(html.Label('Risk-Free Rate'), width=2),
        dbc.Col(dcc.Slider(
            id='risk-free-rate-slider',
            min=0,
            max=0.1,
            step=0.01,
            value=0.01,
            marks={i / 100: f'{i}%' for i in range(0, 11)}
        ), width=4)
    ], justify='center'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='efficient-frontier'), width=6),
        dbc.Col(dcc.Graph(id='cumulative-returns'), width=6)
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='portfolio-weights'), width=12)
    ])
], fluid=True)

# Define callback to update the graphs and weights based on user input
@app.callback(
    [Output('efficient-frontier', 'figure'),
     Output('cumulative-returns', 'figure'),
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

    # Initialize the investment amount
    initial_investment = 1000

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
            continue

    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    # Check for NaN values
    if results_df.isna().any().any():
        return {}, {}, "Error in portfolio generation."

    # Locate the max Sharpe ratio portfolio
    try:
        max_sharpe_idx = np.argmax(results_df['Sharpe Ratio'])
        max_sharpe_weights = weights_record[max_sharpe_idx]
    except ValueError:
        return {}, {}, "Error finding max Sharpe Ratio."

    # Locate the minimum volatility portfolio
    try:
        min_volatility_idx = np.argmin(results_df['Volatility'])
        min_volatility_weights = weights_record[min_volatility_idx]
    except ValueError:
        return {}, {}, "Error finding min Volatility."

    # Calculate portfolio returns
    try:
        max_sharpe_returns = calculate_portfolio_returns(max_sharpe_weights, daily_returns)
        min_volatility_returns = calculate_portfolio_returns(min_volatility_weights, daily_returns)
    except ValueError:
        return {}, {}, "Error calculating portfolio returns."

    # Adjust returns for the initial investment
    try:
        cumulative_max_sharpe_returns = pd.Series((1 + max_sharpe_returns).cumprod() * initial_investment, index=daily_returns.index)
        cumulative_min_volatility_returns = pd.Series((1 + min_volatility_returns).cumprod() * initial_investment, index=daily_returns.index)
    except ValueError:
        return {}, {}, "Error in cumulative returns calculation."

    # Plotting the efficient frontier without legend
    frontier_trace = go.Scatter(
        x=results_df['Volatility'],
        y=results_df['Return'],
        mode='markers',
        marker=dict(color=results_df['Sharpe Ratio'], colorscale='Viridis', size=5),
        name='Portfolios'
    )

    min_volatility_trace = go.Scatter(
        x=[results_df.loc[min_volatility_idx, 'Volatility']],
        y=[results_df.loc[min_volatility_idx, 'Return']],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='diamond'),
        name='Min Volatility Portfolio'
    )

    max_sharpe_trace = go.Scatter(
        x=[results_df.loc[max_sharpe_idx, 'Volatility']],
        y=[results_df.loc[max_sharpe_idx, 'Return']],
        mode='markers',
        marker=dict(color='red', size=10, symbol='diamond'),
        name='Max Sharpe Portfolio'
    )

    efficient_frontier_figure = go.Figure(data=[frontier_trace, min_volatility_trace, max_sharpe_trace])
    efficient_frontier_figure.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Return',
        showlegend=False
    )

    # Plotting cumulative returns
    cumulative_returns_figure = go.Figure()
    cumulative_returns_figure.add_trace(go.Scatter(x=cumulative_max_sharpe_returns.index, y=cumulative_max_sharpe_returns, mode='lines', name='Max Sharpe Portfolio'))
    cumulative_returns_figure.add_trace(go.Scatter(x=cumulative_min_volatility_returns.index, y=cumulative_min_volatility_returns, mode='lines', name='Min Volatility Portfolio'))
    cumulative_returns_figure.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # Prepare the portfolio weights table
    weights_df = pd.DataFrame({
        'Stock': tickers,
        'Min Volatility Portfolio': min_volatility_weights,
        'Max Sharpe Portfolio': max_sharpe_weights
    })

    weights_table = dash_table.DataTable(
        data=weights_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in weights_df.columns],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    )

    return efficient_frontier_figure, cumulative_returns_figure, weights_table

# Function to load stock data
def load_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    daily_returns = stock_data.pct_change().dropna()
    return stock_data, daily_returns

# Function to calculate portfolio performance
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualize the return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualize the volatility
    return portfolio_return, portfolio_volatility

# Function to calculate portfolio returns
def calculate_portfolio_returns(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    return portfolio_returns


# Run the app. Use the port 8050
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
