import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

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

    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'Tech Stocks', 'value': 'AAPL MSFT GOOGL AMZN META'},
            {'label': 'Finance Stocks', 'value': 'JPM BAC GS MS'},
            {'label': 'Healthcare Stocks', 'value': 'JNJ PFE MRK ABBV'}
        ],
        value='AAPL MSFT GOOGL AMZN META'
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

    dcc.Graph(id='efficient-frontier'),
    dcc.Graph(id='cumulative-returns'),

    html.Div(id='portfolio-weights')
])

# Define callback to update the graphs and weights based on user input
@app.callback(
    [Output('efficient-frontier', 'figure'),
     Output('cumulative-returns', 'figure'),
     Output('portfolio-weights', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('risk-free-rate-slider', 'value')]
)
def update_graphs(selected_stocks, risk_free_rate):
    # Load stock data
    tickers = selected_stocks.split()
    stock_data, daily_returns = load_stock_data(tickers, '2018-07-12', '2024-07-12')

    # Generate random portfolios
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(stock_data.columns))
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, daily_returns)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio

    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    # Identify the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
    max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]

    # Identify the portfolio with the minimum volatility
    min_volatility_idx = results_df['Volatility'].idxmin()
    min_volatility_portfolio = results_df.loc[min_volatility_idx]
    min_volatility_weights = weights_record[min_volatility_idx]

    # Calculate historical returns for the optimal portfolios
    max_sharpe_returns = calculate_portfolio_returns(max_sharpe_weights, daily_returns)
    min_volatility_returns = calculate_portfolio_returns(min_volatility_weights, daily_returns)

    cumulative_max_sharpe_returns = (1 + max_sharpe_returns).cumprod()
    cumulative_min_volatility_returns = (1 + min_volatility_returns).cumprod()

    # Efficient Frontier Plot
    efficient_frontier_figure = {
        'data': [
            go.Scatter(
                x=results_df['Volatility'],
                y=results_df['Return'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results_df['Sharpe Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                ),
                name='Portfolios'
            ),
            go.Scatter(
                x=[max_sharpe_portfolio['Volatility']],
                y=[max_sharpe_portfolio['Return']],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name='Max Sharpe Ratio'
            ),
            go.Scatter(
                x=[min_volatility_portfolio['Volatility']],
                y=[min_volatility_portfolio['Return']],
                mode='markers',
                marker=dict(color='blue', size=10, symbol='star'),
                name='Min Volatility'
            )
        ],
        'layout': go.Layout(
            title='Efficient Frontier',
            xaxis=dict(title='Volatility (Risk)'),
            yaxis=dict(title='Return'),
            hovermode='closest'
        )
    }

    # Cumulative Returns Plot
    cumulative_returns_figure = {
        'data': [
            go.Scatter(
                x=cumulative_max_sharpe_returns.index,
                y=cumulative_max_sharpe_returns,
                mode='lines',
                name='Max Sharpe Ratio Portfolio'
            ),
            go.Scatter(
                x=cumulative_min_volatility_returns.index,
                y=cumulative_min_volatility_returns,
                mode='lines',
                name='Min Volatility Portfolio'
            )
        ],
        'layout': go.Layout(
            title='Cumulative Returns of Optimal Portfolios',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Cumulative Return')
        )
    }

    # Display Portfolio Weights
    weights_table = html.Table([
        html.Thead(html.Tr([html.Th("Stock"), html.Th("Max Sharpe Ratio Weights"), html.Th("Min Volatility Weights")])),
        html.Tbody([
            html.Tr([html.Td(stock), html.Td(f"{weight:.2%}"), html.Td(f"{min_volatility_weights[idx]:.2%}")])
            for idx, (stock, weight) in enumerate(zip(tickers, max_sharpe_weights))
        ])
    ])

    return efficient_frontier_figure, cumulative_returns_figure, weights_table

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)

