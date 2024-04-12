import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from datetime import date
import yfinance as yf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from classical_po import ClassicalPO
from scipy.optimize import minimize
import optuna
import ml_collections
import datetime
from pprint import pprint
from statsmodels.tsa.filters.hp_filter import hpfilter
optuna.logging.set_verbosity(optuna.logging.WARNING)
from qiskit_optimization.converters import QuadraticProgramToQubo

from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
from qiskit_optimization import QuadraticProgram
from docplex.mp.model import Model


from qiskit_optimization.converters.quadratic_program_to_qubo import (
    QuadraticProgramToQubo,
)

from qiskit_optimization.translators import from_docplex_mp


seed = 42
cfg = ml_collections.ConfigDict()


st.set_page_config(page_title="PilotProject", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Pilot Project on Portfolio Optimisation ")
#st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)


assets_input = st.multiselect('Enter stock ticker symbols separated by commas, e.g., AAPL,GOOGL,MSFT', 
                             ['BHARTIARTL.NS', 'HDFCBANK.NS','HINDUNILVR.NS','ICICIBANK.NS','INFY.NS','ITC.NS','LT.NS','RELIANCE.NS','SBIN.NS','TCS.NS'],
                              ['BHARTIARTL.NS', 'HDFCBANK.NS','HINDUNILVR.NS','ICICIBANK.NS','INFY.NS','ITC.NS','LT.NS','RELIANCE.NS','SBIN.NS','TCS.NS'] )

print(type(assets_input))
# User input for start and end dates
start_date = st.date_input('Start date of the Portfolio:', datetime.date(2023, 2, 1))
end_date = st.date_input('End date of the Portfolio:',  datetime.date(2024, 2, 26)  )
# print(start_date)
# print(end_date)

# Button to trigger data download
if st.button('Next'):
    # tickers = assets_input  # Remove spaces and split by comma
    # if tickers and tickers[0]:  # Check if there's at least one ticker
    #     # Initialize an empty DataFrame to hold closing prices
    #     close_prices = pd.DataFrame()
        
    #     for ticker in tickers:
            # Downloading the stock data
            # data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # if not data.empty:
            #     # Extracting the closing prices and adding them to the DataFrame
            # else:
            #     st.write(f"No data found for {ticker} with the selected date range.")
    
    
    # Displaying the closing prices
    # closing_prices_df.to_csv('stock_closing_prices.csv')
    # stock_closing_prices = pd.read_csv('stock_closing_prices.csv')
    stock_closing_prices_3y = pd.read_csv('stock_closing_prices.csv', parse_dates=['Date'])
    stock_closing_prices_3y = stock_closing_prices_3y.reset_index(drop=True) #.set_index(['Date'])
    # st.markdown("**Closing Prices:**")
    # st.dataframe(stock_closing_prices_3y)

# Filter DataFrame rows based on date range

    
    # date issue
    
    print(stock_closing_prices_3y)

        # Convert 'Date' column to datetime format
    print(stock_closing_prices_3y['Date'])
    stock_closing_prices_3y['Date'] = pd.to_datetime(stock_closing_prices_3y['Date'])

    # Set 'Date' column as the index
    stock_closing_prices_3y = stock_closing_prices_3y.set_index('Date')

    # Filter DataFrame rows based on date range
    closing_prices_df = stock_closing_prices_3y.loc[start_date:end_date]
    closing_prices_df = closing_prices_df[assets_input]
    
    stock_closing_prices_3y = stock_closing_prices_3y[assets_input]


    # closing_prices_df = stock_closing_prices_3y


    # closing_prices_df = stock_closing_prices_3y.loc[start_date:end_date]
    st.markdown("** Adj Closing Prices:**")
    st.dataframe(closing_prices_df)

    
    class cfg:
        hpfilter_lamb = 6.25
        q = 1.0
        fmin = 0.001
        fmax = 0.5
        num_stocks = len(closing_prices_df.columns)
    
    for s in closing_prices_df.columns:
        cycle, trend = hpfilter(closing_prices_df[s], lamb=cfg.hpfilter_lamb)
        closing_prices_df[s] = trend
    
    log_returns = np.log(closing_prices_df) - np.log(closing_prices_df.shift(1))
    null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
    drop_stocks = closing_prices_df.columns[null_indices]
    log_returns = log_returns.drop(columns=drop_stocks)
    log_returns = log_returns.dropna()
    tickers = log_returns.columns
    cfg.num_stocks = len(tickers)
    mu = log_returns.mean().to_numpy() * 252
    sigma = log_returns.cov().to_numpy() * 252

    
    plt.figure(figsize = (4,4))
    for s in log_returns.columns:
        plt.plot(closing_prices_df[s], label=s)
    legend_fontsize = 8
    plt.legend(loc="upper center", bbox_to_anchor=(2.0, 1.1), fancybox=True, shadow=True, ncol=4, fontsize=legend_fontsize)
    plt.xlabel("Days")
    plt.ylabel("Stock Prices")
    plt.title("Stock Prices Over Time")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    cfg.kappa = cfg.num_stocks
    
    def create_problem(mu: np.array, sigma: np.array, qiskit_budget: int, alpha: float) -> QuadraticProgram:
        """Solve the quadratic program using docplex."""

        mdl = Model()
        x = [mdl.binary_var("x%s" % i) for i in range(len(sigma))]

        objective = mdl.sum([mu[i] * x[i] for i in range(len(mu))])
        objective -= alpha * mdl.sum(
            [sigma[i, j] * x[i] * x[j] for i in range(len(mu)) for j in range(len(mu))]
        )
        # objective = 
        mdl.maximize(objective)
        cost = mdl.sum(x)
        mdl.add_constraint(cost == qiskit_budget)

        qp = from_docplex_mp(mdl)
        return qp
    
    def objective_mvo_miqp(trial, _mu, _sigma):
        cpo = ClassicalPO(_mu, _sigma, cfg)
        cpo.cfg.gamma = trial.suggest_float('gamma', 0.0, 1.5)
        res = cpo.mvo_miqp()
        mvo_miqp_res = cpo.get_metrics(res['w'])
        del cpo
        return mvo_miqp_res['sharpe_ratio']
    
    study_mvo_miqp = optuna.create_study(
        study_name='classical_mvo_miqp',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        load_if_exists=True)
    
    
    mu = np.array([3.418, 2.0913, 6.2415, 4.4436, 10.892, 3.4051])
    sigma = np.array(
        [
            [1.07978412, 0.00768914, 0.11227606, -0.06842969, -0.01016793, -0.00839765],
            [0.00768914, 0.10922887, -0.03043424, -0.0020045, 0.00670929, 0.0147937],
            [0.11227606, -0.03043424, 0.985353, 0.02307313, -0.05249785, 0.00904119],
            [-0.06842969, -0.0020045, 0.02307313, 0.6043817, 0.03740115, -0.00945322],
            [-0.01016793, 0.00670929, -0.05249785, 0.03740115, 0.79839634, 0.07616951],
            [-0.00839765, 0.0147937, 0.00904119, -0.00945322, 0.07616951, 1.08464544],
        ]
    )
    qiskit_budget = len(mu)//2
    alpha = 1.0
    qp = create_problem(mu, sigma, qiskit_budget, alpha)
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)


    sampler_dw = DWaveSampler(solver='Advantage_system4.1', token = 'DEV-0fb34d846c85d1e57c7974c2679751c459594c1d')
    sampler_qa = EmbeddingComposite(sampler_dw)

    bqm_qubo = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)

    embedded_sampler = EmbeddingComposite(sampler_dw)
    result_using_dwave = embedded_sampler.sample(bqm_qubo, label="example_qp", num_reads=5000)
    print("dwave result is ", result_using_dwave)
    st.text(result_using_dwave)
    
    
    study_mvo_miqp.optimize(lambda trial: objective_mvo_miqp(trial, mu, sigma), n_trials=25-len(study_mvo_miqp.trials), n_jobs=1)
    trial_mvo_miqp = study_mvo_miqp.best_trial
    cpo = ClassicalPO(mu, sigma, cfg)
    cpo.cfg.gamma = 1.9937858736079478
    res = cpo.mvo_miqp()
    weights = res['w']
    stock_dict = dict(zip(tickers, np.around(weights, 5)))
    stock_dict = {i: j for i, j in stock_dict.items() if j != 0}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Weights Allocated by the Algorithm:**")
        #st.text(f"{'Stock':<25}{'Weights(%)':>15}")
        for stock, value in stock_dict.items():
            percentage = round(value * 100, 2)  # Multiply by 100 and round off to 2 decimal places
            st.text(f"{stock:<25}{percentage:>15}")

    #st.write(stock_dict)

        st.markdown("**Returns and Risk of the portfolio:**")
        mvo_miqp_res = cpo.get_metrics(weights)
        for metric, value in mvo_miqp_res.items():
            if metric in ['returns', 'risk']:
                display_value = round(value*100,2)
            else:
                display_value = round(value, 2)
            st.text(f"{metric:<25}{display_value:>15}")
    #st.write(mvo_miqp_res)
            
    with col2:
        weights_axis = {
        'BHARTIARTL.NS': 0.0523,
        'HDFCBANK.NS':  0.0936,
        'HINDUNILVR.NS': 0.1491,
        'ICICIBANK.NS': 0.0552,
        'INFY.NS':  0.0841,
        'ITC.NS': 0.0253,
        'LT.NS':   0.1588,
        'RELIANCE.NS':  0.1449,
        'SBIN.NS': 0.0342,
        'TCS.NS': 0.2025}

        st.markdown("**Weights given in the Attribution Report:**")
    #st.text(f"{'Stock':<25}{'Weights(%)':>15}")
        for stock, value in weights_axis.items():
            percentage = round(value * 100, 2)  # Multiply by 100 and round off to 2 decimal places
            st.text(f"{stock:<25}{percentage:>15}")
    #st.write(weights_axis)
            
        weights_axis_array = np.array([0.0523, 0.0936, 0.1491, 0.0552, 0.0841, 0.0253, 0.1588, 0.1449, 0.0342, 0.2025])
        st.markdown("**Returns and Risk of the Attribution Report:**")
        mvo_miqp_axis = cpo.get_metrics(weights_axis_array)
        for metric, value in mvo_miqp_axis.items():
            if metric in ['returns', 'risk']:
                display_value = round(value*100,2)
            else:
                display_value = round(value, 2)
            st.text(f"{metric:<25}{display_value:>15}")
            
        

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'DarkOrchid', 'DeepPink', 'Maroon', 'MistyRose', 'Olive', 'Salmon' ]
    fig = go.Figure(data=[go.Pie(labels=list(stock_dict.keys()), values=list(stock_dict.values()), hole=.3)])
    fig.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    st.markdown("**Pie Chart of Stock Weights Allocated by the Algorithm:**")
    st.plotly_chart(fig)

    sector_weights = {}
    sectors = {'Information Technology': ['INFY.NS', 'TCS.NS'], 'Financials': ['HDFCBANK.NS', 'SBIN.NS', 'ICICIBANK.NS'],
    'Consumer Staples': ['HINDUNILVR.NS', 'ITC.NS'], 'Industrials':['LT.NS'], 'Energy': ['RELIANCE.NS'], 'Communication Services':['BHARTIARTL.NS']}
    for stock, weight in stock_dict.items():
        for sector, stocks_in_sector in sectors.items():
            if stock in stocks_in_sector:
                sector_weights.setdefault(sector, 0)
                sector_weights[sector] += weight
        
    keys = sector_weights.keys()
    values_sector = sector_weights.values()
    fig_sector = go.Figure(data=[go.Pie(labels=list(keys),values=list(values_sector), hole=.3)])
    fig_sector.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
    st.markdown("**Pie Chart of Sector Weights Allocated by the Algorithm:**")
    st.plotly_chart(fig_sector)

    
    
    #st.markdown("**Returns and Risk of the Attribution Report:**")
    #st.write(mvo_miqp_axis)

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'DarkOrchid', 'DeepPink', 'Maroon', 'MistyRose', 'Olive', 'Salmon' ]
    fig_axis = go.Figure(data=[go.Pie(labels=list(weights_axis.keys()), values=list(weights_axis.values()), hole=.3)])
    fig_axis.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    st.markdown("**Pie Chart of Stock Weights given in the Attribution Report:**")
    st.plotly_chart(fig_axis)

    sector_weights_axis= {}
    sectors = {'Information Technology': ['INFY.NS', 'TCS.NS'], 'Financials': ['HDFCBANK.NS', 'SBIN.NS', 'ICICIBANK.NS'],
    'Consumer Staples': ['HINDUNILVR.NS', 'ITC.NS'], 'Industrials':['LT.NS'], 'Energy': ['RELIANCE.NS'], 'Communication Services':['BHARTIARTL.NS']}
    for stock, weight in weights_axis.items():
        for sector, stocks_in_sector in sectors.items():
            if stock in stocks_in_sector:
                sector_weights_axis.setdefault(sector, 0)
                sector_weights_axis[sector] += weight

    keys_axis = sector_weights_axis.keys()
    values_sector_axis = sector_weights_axis.values()
    fig_sector_axis = go.Figure(data=[go.Pie(labels=list(keys_axis),values=list(values_sector_axis), hole=.3)])
    fig_sector_axis.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
    st.markdown("**Pie Chart of Sector Weights given in the Attribution Report:**")
    st.plotly_chart(fig_sector_axis)
    
# data = 13 months data file                 
# data = pd.read_csv('stock_closing_prices.csv', parse_dates=['Date'])
# stock_closing_prices = pd.read_csv('stock_closing_prices.csv')  data = stocks_closing_parse 


    first_row_prices = data.iloc[0, 0:]
    investment_values = first_row_prices * 100
    total_investment_amount = investment_values.sum()
    st.markdown("**Total Investment Amount (in rupees):**")
    st.write(total_investment_amount)
    investment_per_stock = {stock: total_investment_amount * weight for stock, weight in weights_axis.items()}
    optimal_stocks_to_buy = {stock: investment // stock_closing_prices.loc[0, stock] for stock, investment in investment_per_stock.items()}
    st.markdown("**Optimal Number of Stocks to buy (weights given in the attribution report):**")
    #st.write(optimal_stocks_to_buy)
    #st.text(f"{'Stock':<25}{'Stocks to buy':>15}")
    for stock, value in optimal_stocks_to_buy.items():
        st.text(f"{stock:<25}{value:>15}")
        
# stock_closing_prices = pd.read_csv('stock_closing_prices.csv')
# 

    portfolio_values = stock_closing_prices.apply(lambda row: sum(
    row[stock] * optimal_stocks_to_buy[stock] for stock in optimal_stocks_to_buy), axis=1)
    stock_closing_prices['Portfolio Value'] = portfolio_values
    st.markdown("**Portfolio Value of Attribution Report:**")
    st.dataframe(stock_closing_prices.tail())

    stock_closing_prices['Date'] = pd.to_datetime(stock_closing_prices['Date'])
    fig = px.line(stock_closing_prices, x='Date', y='Portfolio Value', 
                    title='Portfolio Value Over Time - Attribution Report',
                    labels={'Portfolio Value': 'Portfolio Value(AMAR)'},
                    markers=True, color_discrete_sequence=['red'],)
    
    fig.update_traces(name='Portfolio Value AMAR', showlegend=True)
    fig.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (in rupees)', autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

    df = pd.read_csv('stock_closing_prices.csv', usecols=range(11))
    investment_per_stock_us = {stock: total_investment_amount * weight for stock, weight in stock_dict.items()}
    optimal_stocks_to_buy_us = {stock: investment // df.loc[0, stock] for stock, investment in investment_per_stock_us.items()}
    st.markdown("**Optimal Number Stocks to buy (weights given by the Algorithm):**")
    #st.write(optimal_stocks_to_buy_us)
    #st.text(f"{'Stock':<25}{'Stocks to buy':>15}")
    for stock, value in optimal_stocks_to_buy_us.items():
        st.text(f"{stock:<25}{value:>15}")

    portfolio_values = df.apply(lambda row: sum(
    row[stock] * optimal_stocks_to_buy_us[stock] for stock in optimal_stocks_to_buy_us), axis=1)
    df['Portfolio Value'] = portfolio_values
    st.markdown("**Portfolio Value given by the Algorithm:**")
    st.dataframe(df.tail())

    df['Date'] = pd.to_datetime(df['Date'])
    fig_port = px.line(df, x='Date', y='Portfolio Value', 
                    title='Portfolio Value Over Time - Qkrishi',
                    labels={'Portfolio Value': 'Portfolio Value(Qkrishi)'},
                    markers=True, color_discrete_sequence=['blue'])
    
    fig_port.update_traces(name='Portfolio Value Qkrishi', showlegend=True)
    fig_port.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (in rupees)', autosize=False, width=1000, height=600)
    st.plotly_chart(fig_port)


    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(x=stock_closing_prices['Date'], 
                    y=stock_closing_prices['Portfolio Value'], 
                    mode='lines+markers', 
                    name='Portfolio Value AMAR', 
                    line=dict(color='red')))
    
    fig_compare.add_trace(go.Scatter(x=df['Date'], 
                    y=df['Portfolio Value'], 
                    mode='lines+markers', 
                    name='Portfolio Value Qkrishi', 
                    line=dict(color='blue')))  
    
    fig_compare.update_layout(title='Portfolio Value Over Time - Comparison',
            xaxis_title='Date', 
            yaxis_title='Portfolio Value (in rupees)',
            autosize=False, 
            width=1000, 
            height=600,)
    
    st.plotly_chart(fig_compare)


    df2 = pd.read_csv('rebalancing_output.csv')

    fig_port_compare = go.Figure() 

    #fig_port_compare.add_trace(go.Scatter(x=df2['Date'], y=df2['Rebalance Value'], 
                                            #mode='lines+markers', name='Rebalance Value', line=dict(color='green'), showlegend=True))
    
    fig_port_compare.add_trace(go.Scatter(x=df2['Date'], y=df2['AMAR Value'], 
                                            mode='lines+markers', name='AMAR Value', line=dict(color='red'), showlegend=True))
    
    fig_port_compare.add_trace(go.Scatter(x=df2['Date'], y=df2['Qkrishi Value'], 
                                            mode='lines+markers', name='Qkrishi Value', line=dict(color='blue'), showlegend=True))
    
    fig_port_compare.update_layout(title='Rebalanced Values Over Time',
            xaxis_title='Date', 
            yaxis_title='Final Value',
            autosize=False, 
            width=1000, 
            height=600,)
    
    st.plotly_chart(fig_port_compare)
    
    
    # else:
    #     st.write("No closing price data found for the selected tickers and date range.")
# else:
#     st.write("Please enter valid stock ticker symbols separated by commas.")


