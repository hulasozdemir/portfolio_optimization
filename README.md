# Portfolio Optimization Dashboard

This project demonstrates portfolio optimization using Modern Portfolio Theory (MPT). It analyzes historical stock data, performs exploratory data analysis, calculates risk and return metrics, and optimizes portfolios to identify the maximum Sharpe ratio and minimum volatility portfolios. The project also includes an interactive dashboard for visualization.

## Features

- Data collection from Yahoo Finance
- Exploratory Data Analysis (EDA) with visualizations
- Calculation of risk and return metrics (Sharpe ratio, Sortino ratio, Treynor ratio)
- Portfolio optimization using Modern Portfolio Theory
- Interactive dashboard to visualize the efficient frontier, cumulative returns, and portfolio weights

## Dashboard
Click [here](https://portfoliooptimization-production.up.railway.app) to go to the dasboard.

## Installation

1. **Clone the repository**:
   ```bash
   git clone origin https://github.com/hulasozdemir/portfolio_optimization.git
   cd portfolio_optimization

2. **Build the Docker image**:

```bash
docker build -t portfolio-optimization-dashboard .
```

3. **Run the docker image**:

```bash
docker run -p 8050:8050 portfolio-optimization-dashboard
```

4. **View the dasboard**:
Go to `http://localhost:8050/` to view the dashboard.


# Data source:
Historical stock price data from Yahoo Finance
