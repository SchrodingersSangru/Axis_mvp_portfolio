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
from SinglePeriod import SinglePeriod
from classical_po import ClassicalPO
from scipy.optimize import minimize
from SinglePeriod import SinglePeriod
import optuna
from itertools import product
import dimod
import datetime
from dimod import quicksum, Integer, Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
#from tabulate import tabulate
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from dwave.samplers import SimulatedAnnealingSampler
import ml_collections
from statsmodels.tsa.filters.hp_filter import hpfilter
optuna.logging.set_verbosity(optuna.logging.WARNING)

# seed = 42
# cfg = ml_collections.ConfigDict()

# st.set_page_config(page_title="PilotProject", page_icon=":chart_with_upwards_trend:", layout="wide")
# st.title("Pilot Project on Portfolio Optimisation :chart_with_upwards_trend: ")
# #st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)

# ticker_input = st.text_input('Enter stock ticker symbols separated by commas, e.g., AAPL,GOOGL,MSFT')
# start_date = st.date_input('Start date of the Portfolio:')
# end_date = st.date_input('End date of the Portfolio:')


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
end_date = st.date_input('End date of the Portfolio:',  datetime.date(2024, 1, 31)  )


if st.button('Next'):
    if assets_input == ['BHARTIARTL.NS', 'HDFCBANK.NS','HINDUNILVR.NS','ICICIBANK.NS','INFY.NS','ITC.NS','LT.NS','RELIANCE.NS','SBIN.NS','TCS.NS']:
        
        stock_closing_prices_3y = pd.read_csv('stock_closing_prices.csv', parse_dates=['Date'])
        stock_closing_prices_3y = stock_closing_prices_3y.reset_index(drop=True) #.set_index(['Date'])
        # st.markdown("**Closing Prices:**")
        # st.dataframe(stock_closing_prices_3y)

        # Filter DataFrame rows based on date range

        
        # date issue
        
        # print(stock_closing_prices_3y)

            # Convert 'Date' column to datetime format
        # print(stock_closing_prices_3y['Date'])
        stock_closing_prices_3y['Date'] = pd.to_datetime(stock_closing_prices_3y['Date'])

        # Set 'Date' column as the index
        stock_closing_prices_3y = stock_closing_prices_3y.set_index('Date')

        # Filter DataFrame rows based on date range
        closing_prices_df = stock_closing_prices_3y.loc[start_date:end_date]
        st.write(closing_prices_df)
        closing_prices_df = closing_prices_df[assets_input]
        
        stock_closing_prices_3y = stock_closing_prices_3y[assets_input]


        # closing_prices_df = stock_closing_prices_3y

        # closing_prices_df = stock_closing_prices_3y.loc[start_date:end_date]
        #st.markdown("** Adj Closing Prices:**")
        #st.dataframe(closing_prices_df)
        #print(closing_prices_df)
        
        
        # stock_closing_prices_3y = closing_prices_df.reset_index(drop=True)
        
        
        class cfg:
            hpfilter_lamb = 6.25
            q = 1.0
            fmin = 0.001
            fmax = 0.5
            num_stocks = len(closing_prices_df.columns)

        stock_prices_for_algo = closing_prices_df.reset_index(drop=True)
        
        for s in closing_prices_df.columns:
            cycle, trend = hpfilter(closing_prices_df[s], lamb=cfg.hpfilter_lamb)
            closing_prices_df[s] = trend
        
        log_returns = np.log(stock_prices_for_algo) - np.log(stock_prices_for_algo.shift(1))
        null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
        drop_stocks = stock_prices_for_algo.columns[null_indices]
        log_returns = log_returns.drop(columns=drop_stocks)
        log_returns = log_returns.dropna()
        tickers = log_returns.columns
        cfg.num_stocks = len(tickers)
        mu = log_returns.mean().to_numpy() * 252
        sigma = log_returns.cov().to_numpy() * 252

        
        plt.figure(figsize = (4,4))
        for s in log_returns.columns:
            plt.plot(stock_prices_for_algo[s], label=s)
        legend_fontsize = 8
        plt.legend(loc="upper center", bbox_to_anchor=(2.0, 1.1), fancybox=True, shadow=True, ncol=4, fontsize=legend_fontsize)
        plt.xlabel("Days")
        plt.ylabel("Stock Prices")
        plt.title("Stock Prices Over Time")
        # plt.tight_layout()  this is givng problem 
        st.pyplot(plt)
        plt.close()

        cfg.kappa = cfg.num_stocks
        
        
        
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
            
            st.markdown("**Returns and Risk of the portfolio:**")
            mvo_miqp_res = cpo.get_metrics(weights)
            for metric, value in mvo_miqp_res.items():
                if metric in ['returns', 'risk']:
                    display_value = round(value*100,2)
                else:
                    display_value = round(value, 2)
                st.text(f"{metric:<25}{display_value:>15}")

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

        # -------- AXIS REQUIREMENTS STARTS HERE -------

        # POINT (a)
        st.markdown('**Line chart of Port against the Benchmark (Rebased to 100 for initial date)**')

        hdfc50 = pd.read_csv('HDFCNIFETF.NS.csv')
        hdfc50['HDFCNIFETF.NS'] = hdfc50['Close']
        hdfc50 = hdfc50[['Date', 'HDFCNIFETF.NS']]
        total_investment_amountt = 1604500.43
        day1_price = hdfc50['HDFCNIFETF.NS'].iloc[0]
        optimal_stocks_to_buy_hdfcnifty50 = total_investment_amountt // day1_price
        portfolio_values_nifty50 = hdfc50['HDFCNIFETF.NS']*optimal_stocks_to_buy_hdfcnifty50
        hdfc50['Portfolio Value'] = portfolio_values_nifty50
        hdfc50['Return'] = hdfc50['Portfolio Value'] * 100 / hdfc50['Portfolio Value'][0]
        hdfc50['Date'] = pd.to_datetime(hdfc50['Date'], format='%d-%m-%Y')
        stock_closing_prices_hdfc = pd.read_csv('stock_closing_prices.csv')
        weights_hdfc = {
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
        
        investment_per_stock_hdfc = {stock: total_investment_amountt * weight for stock, weight in weights_hdfc.items()}
        optimal_stocks_hdfc = {stock: investment // stock_closing_prices_hdfc.loc[0, stock] for stock, investment in investment_per_stock_hdfc.items()}
        portfolio_valuesss = stock_closing_prices_hdfc.apply(lambda row: sum(row[stock] * optimal_stocks_hdfc[stock] for stock in optimal_stocks_hdfc), axis=1)
        stock_closing_prices_hdfc['Portfolio Value'] = portfolio_valuesss
        stock_closing_prices_hdfc['Return'] = stock_closing_prices_hdfc['Portfolio Value'] * 100 / stock_closing_prices_hdfc['Portfolio Value'][0]

        fig_compare_hdfc = go.Figure()

        fig_compare_hdfc.add_trace(go.Scatter(x= stock_closing_prices_hdfc['Date'], 
                    y=  stock_closing_prices_hdfc['Return'],
                    mode='lines+markers', 
                    name='Return AMAR', 
                    line=dict(color='red')))
    
        fig_compare_hdfc.add_trace(go.Scatter(x=hdfc50['Date'], 
                    y=hdfc50['Return'], 
                    mode='lines+markers', 
                    name='Return HDFCNIFTY50ETF', 
                    line=dict(color='blue')))
    
        fig_compare_hdfc.update_layout(title='Return Over Time',
                xaxis_title='Date', 
                yaxis_title='Return',
                autosize=False, 
                width=1000, 
                height=600,)
        
        st.plotly_chart(fig_compare_hdfc)

        # POINT (b)
        st.markdown('**Portfolio Weight vs Benchmark Weight**')
        port = {'Information Technology': '28.66', 'Financials':'18.29', 'Consumer Staples':'17.44', 'Industrials':'15.88', 'Energy':'14.49', 'Communication Services':'5.23'}
        bench = {'Information Technology': '13.64', 'Financials':'36.37', 'Consumer Staples':'9.38', 'Industrials':'5.40', 'Energy':'11.89', 'Communication Services':'2.63'}
        common_keys = set(port.keys()) & set(bench.keys())
        for key in common_keys:
            port_value = float(port[key])
            bench_value = float(bench[key])
            difference = port_value - bench_value
            if difference > 0:
                st.text(f"{key}: Overweight by {abs(difference)}")
            elif difference < 0:
                st.text(f"{key}: Underweight by {abs(difference)}")
            else:
                st.text(f"{key}: Neutral")

        # POINT (c)
        st.markdown('**Sebi Classification-wise Weights**')
        bench_weights = {'HDFC Bank Ltd': 11.48,
         'Reliance Industries Ltd': 9.96,
         'ICICI Bank Ltd': 8.11,
         'Infosys Ltd':5.09,
         'Larsen & Toubro Ltd':4.27,
         'Tata Consultancy Services Ltd':3.89,
         'ITC Ltd':3.88,
         'Bharti Airtel Ltd':3.45,
         'Axis Bank Ltd':3.32,
         'State Bank of India':3.18,
         'Kotak Mahindra Bank Ltd':2.40 ,
         'Mahindra & Mahindra Ltd' :2.07 ,
         'Hindustan Unilever Ltd' :2.00 ,
         'Bajaj Finance Ltd' :1.94 ,
         'Tata Motors Ltd' :1.78 ,
         'NTPC Ltd' :1.73 ,
         'Maruti Suzuki India Ltd' :1.70 ,
         'Sun Pharmaceuticals Industries Ltd' : 1.63 ,
         'Titan Co Ltd' :1.50 ,
         'HCL Technologies Ltd' : 1.45,
         'Power Grid Corp Of India Ltd' :1.38 ,
         'Tata Steel Ltd' : 1.36 ,
         'Asian Paints Ltd' : 1.30,
         'UltraTech Cement Ltd' : 1.16,
         'Oil & Natural Gas Corp Ltd' :1.11 ,
         'Coal India Ltd' :1.04 ,
         'Bajaj Auto Ltd' :1.01 ,
         'IndusInd Bank Ltd' :1.01 ,
         'Adani Ports & Special Economic Zone Ltd' : 0.98 ,
         'Hindalco Industries Ltd' : 0.94 ,
         'Nestle India Ltd' : 0.90 ,
         'Grasim Industries Ltd' : 0.89 ,
         'Bajaj Finserv Ltd' : 0.88 ,
         'JSW Steel Ltd' : 0.84 ,
         'Tech Mahindra Ltd' : 0.81 ,
         'Adani Enterprises Ltd' : 0.80 ,
         'Dr Reddys Laboratories Ltd' : 0.76 ,
         'Cipla Ltd' : 0.74 ,
         'Shriram Finance Ltd' : 0.71 ,
         'Tata Consumer Products Ltd' : 0.70 ,
         'Wipro Ltd' : 0.65 ,
         'SBI Life Insurance Company Limited' : 0.65 ,
         'Eicher Motors Ltd' : 0.63 ,
         'HDFC Life Insurance Company Limited' : 0.62 ,
         'Apollo Hospitals Enterprise Ltd' : 0.60 ,
         'Hero MotoCorp Ltd' : 0.59 ,
         'Bharat Petroleum Corp Ltd' : 0.58 ,
         'Britannia Industries Ltd' : 0.57 ,
         'Divis Laboratories Ltd' : 0.51 ,
         'LTIMindtree Ltd' : 0.43 }

        port_weights = {'Tata Consultancy Services Ltd':20.25,
        'Infosys Ltd':8.41,
        'HDFC Bank Ltd':9.36,
        'ICICI Bank Ltd':5.52,
        'State Bank of India':3.42,
        'Hindustan Unilever Ltd':14.91,
        'ITC Ltd':2.53,
        'Larsen & Toubro Ltd':15.88,
        'Reliance Industries Ltd': 14.49,
        'Bharti Airtel Ltd': 5.23}
    
        large_cap = ['HDFC Bank Ltd',
         'Reliance Industries Ltd',
         'ICICI Bank Ltd',
         'Infosys Ltd',
         'Larsen & Toubro Ltd',
         'Tata Consultancy Services Ltd',
         'ITC Ltd',
         'Bharti Airtel Ltd',
         'Axis Bank Ltd',
         'State Bank of India',
         'Kotak Mahindra Bank Ltd',
         'Mahindra & Mahindra Ltd'  ,
         'Hindustan Unilever Ltd'  ,
         'Bajaj Finance Ltd'  ,
         'Tata Motors Ltd'  ,
         'NTPC Ltd' ,
         'Maruti Suzuki India Ltd' ,
         'Sun Pharmaceuticals Industries Ltd'  ,
         'Titan Co Ltd' ,
         'HCL Technologies Ltd',
         'Power Grid Corp Of India Ltd' ,
         'Tata Steel Ltd' ,
         'Asian Paints Ltd',
         'UltraTech Cement Ltd' ,
         'Oil & Natural Gas Corp Ltd'  ,
         'Coal India Ltd'  ,
         'Bajaj Auto Ltd' ,
         'IndusInd Bank Ltd' ,
         'Adani Ports & Special Economic Zone Ltd' ,
         'Hindalco Industries Ltd'  ,
         'Nestle India Ltd' ,
         'Grasim Industries Ltd'  ,
         'Bajaj Finserv Ltd' ,
         'JSW Steel Ltd'  ,
         'Tech Mahindra Ltd'  ,
         'Adani Enterprises Ltd'  ,
         'Dr Reddys Laboratories Ltd' ,
         'Cipla Ltd',
         'Shriram Finance Ltd' ,
         'Tata Consumer Products Ltd' ,
         'Wipro Ltd' ,
         'SBI Life Insurance Company Limited'  ,
         'Eicher Motors Ltd',
         'HDFC Life Insurance Company Limited',
         'Apollo Hospitals Enterprise Ltd' ,
         'Bharat Petroleum Corp Ltd' ,
         'Britannia Industries Ltd' ,
         'Divis Laboratories Ltd' ,
         'LTIMindtree Ltd']

        mid_cap = ['Hero MotoCorp Ltd']

        large_cap_weight = sum(float(bench_weights.get(company, 0)) for company in large_cap)
        mid_cap_weight = sum(float(bench_weights.get(company, 0)) for company in mid_cap)
        labels = ['Large Cap', 'Mid Cap']
        sizes = [large_cap_weight, mid_cap_weight]
        colors = ['gold', 'DeepPink' ]

        fig_sector_bench = go.Figure(data=[go.Pie(labels=labels,values=sizes, hole=.3)])
        fig_sector_bench.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        st.markdown("**Benchmark Weight Distribution**")
        st.plotly_chart(fig_sector_bench)

        large_cap_weight_port = sum(float(port_weights.get(company, 0)) for company in large_cap)
        mid_cap_weight_port = sum(float(port_weights.get(company, 0)) for company in mid_cap)
        labels_port = ['Large Cap', 'Mid Cap']
        sizes_port = [large_cap_weight_port, mid_cap_weight_port]
        colors = ['gold', 'DeepPink' ]
        fig_sector_port = go.Figure(data=[go.Pie(labels=labels_port,values=sizes_port, hole=.3)])
        fig_sector_port.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        st.markdown("**Portfolio Weight Distribution**")
        st.plotly_chart(fig_sector_port)

        #POINT (e)
        st.markdown('**Top 10 best and Bottom 10 worst performers(Based on Return)**')
        adj_close_df = pd.read_csv(f'stock_closing_prices.csv', usecols=range(11))
        data = adj_close_df.drop(columns=['Date'])
        tickers = sorted(["BHARTIARTL.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"])
        for s in data.columns:
            cycle, trend = hpfilter(data[s], lamb=cfg.hpfilter_lamb)
            data[s] = trend

        log_returns = np.log(data) - np.log(data.shift(1))
        null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
        drop_stocks = data.columns[null_indices]
        log_returns = log_returns.drop(columns=drop_stocks)
        log_returns = log_returns.dropna()
        tickers = log_returns.columns
        cfg.num_stocks = len(tickers)

        mu = log_returns.mean()* 252
        sigma = log_returns.cov().to_numpy() * 252

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('**Bottom 10 Worst**')
            st.write(mu.sort_values())
        with col4:
            st.markdown('**Top 10 Best**')
            st.write(mu.sort_values(ascending=False))

        #POINT (f)
        st.markdown('**Top 10 relative weight, Bottom 10 relative weight**')
        port_weights = {'Tata Consultancy Services Ltd':'20.25',
        'Infosys Ltd':'8.41',
        'HDFC Bank Ltd':'9.36',
        'ICICI Bank Ltd':'5.52',
        'State Bank of India':'3.42',
        'Hindustan Unilever Ltd':'14.91',
        'ITC Ltd':'2.53',
        'Larsen & Toubro Ltd':'15.88',
        'Reliance Industries Ltd': '14.49',
        'Bharti Airtel Ltd': '5.23'}

        benchmark_weights = {'Tata Consultancy Services Ltd':'4.17',
        'Infosys Ltd':'6.07',
        'HDFC Bank Ltd':'11.20',
        'ICICI Bank Ltd':'7.76',
        'State Bank of India':'2.64',
        'Hindustan Unilever Ltd':'2.68',
        'ITC Ltd':'4.49',
        'Larsen & Toubro Ltd':'3.84',
        'Reliance Industries Ltd': '9.89',
        'Bharti Airtel Ltd': '2.63'}

        keys = set(port_weights.keys()) & set(benchmark_weights.keys())
        results = []
        for key in keys:
            port_value = float(port_weights[key])
            bench_value = float(benchmark_weights[key])
            difference = port_value - bench_value
            results.append((key, abs(difference)))
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        sorted_results_bottom = sorted(results, key=lambda x: x[1], reverse=False)
        st.markdown('**Top 10 Relative Weight**')
        st.text(sorted_results)

        st.markdown('**Bottom 10 Relative Weight**')
        st.text(sorted_results_bottom)

        #POINT (g)
        st.markdown('**Top 10 holdings, Bottom 10 holdings (With performance of 1m, 3m, 6m, 1 yr)**')
        stock_data = pd.read_csv('stock_closing_prices.csv')
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
        def func(dataframe, days):
            log_returns = np.log(dataframe) - np.log(dataframe.shift(1))
            null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
            drop_stocks = dataframe.columns[null_indices]
            log_returns = log_returns.drop(columns=drop_stocks)
            log_returns = log_returns.dropna()
            tickers = log_returns.columns
            mu = log_returns.mean() * days
            sigma = log_returns.cov().to_numpy() * days
            return mu.sort_values()
    
        start_date = pd.to_datetime('2023-02-01')
        end_date = pd.to_datetime('2023-03-01')
        filtered_df = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)].drop(columns=['Date'])

        start_date_3 = pd.to_datetime('2023-02-01')
        end_date_3 = pd.to_datetime('2023-05-01')
        filtered_df_3 = stock_data[(stock_data['Date'] >= start_date_3) & (stock_data['Date'] <= end_date_3)].drop(columns=['Date'])

        start_date_6 = pd.to_datetime('2023-02-01')
        end_date_6 = pd.to_datetime('2023-08-01')
        filtered_df_6 = stock_data[(stock_data['Date'] >= start_date_6) & (stock_data['Date'] <= end_date_6)].drop(columns=['Date'])

        start_date_1y = pd.to_datetime('2023-02-01')
        end_date_1y = pd.to_datetime('2024-02-01')
        filtered_df_1y = stock_data[(stock_data['Date'] >= start_date_1y) & (stock_data['Date'] <= end_date_1y)].drop(columns=['Date'])

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown('**One Month Return**')
            st.write(func(filtered_df, 21))
        with col6:
            st.markdown('**Three Month Return**')
            st.write(func(filtered_df_3, 57))
        with col7:
            st.markdown('**Six Month Return**')
            st.write(func(filtered_df_6, 123))
        with col8:
            st.markdown('**One Year Return**')
            st.write(func(filtered_df_1y, 246))
        
        #POINT (h)
        st.markdown('**Top 10 contribution to returns, Bottom 10 contribution to returns**')
        port_weights = {'Tata Consultancy Services Ltd': 20.25,
        'Infosys Ltd': 8.41,
        'HDFC Bank Ltd': 9.36,
        'ICICI Bank Ltd': 5.52,
        'State Bank of India': 3.42,
        'Hindustan Unilever Ltd': 14.91,
        'ITC Ltd': 2.53,
        'Larsen & Toubro Ltd': 15.88,
        'Reliance Industries Ltd': 14.49,
        'Bharti Airtel Ltd': 5.23}

        benchmark_weights = {'Tata Consultancy Services Ltd': 4.17,
        'Infosys Ltd': 6.07,
        'HDFC Bank Ltd': 11.20,
        'ICICI Bank Ltd': 7.76,
        'State Bank of India': 2.64,
        'Hindustan Unilever Ltd': 2.68,
        'ITC Ltd': 4.49,
        'Larsen & Toubro Ltd':3.84,
        'Reliance Industries Ltd': 9.89,
        'Bharti Airtel Ltd': 2.63}

        total_return = {'Tata Consultancy Services Ltd': 19.70,
        'Infosys Ltd': 9.90,
        'HDFC Bank Ltd': -11.60,
        'ICICI Bank Ltd': 25.35,
        'State Bank of India': 46.71,
        'Hindustan Unilever Ltd': -5.16,
        'ITC Ltd':19.20,
        'Larsen & Toubro Ltd':63.59,
        'Reliance Industries Ltd':40.04,
        'Bharti Airtel Ltd': 44.99}

        contribution_return_port = {}
        for key in port_weights:
            contribution_return_port[key] = port_weights[key] * total_return[key]
        st.markdown('**Contribution to return - PORT, Bottom 10**')
        st.text(sorted(contribution_return_port.items(), key=lambda x: x[1]))

        contribution_return_bench = {}
        for key in benchmark_weights:
            contribution_return_bench[key] = benchmark_weights[key] * total_return[key]
        st.markdown('**Contribution to return - BENCH, Bottom 10**')
        st.text(sorted(contribution_return_bench.items(), key=lambda x: x[1]))

        #Penny Stocks
        st.markdown('**Penny Stocks**')
        st.text('There are no penny stocks included our portfolio as all of the listed stocks fall under HDFCNIFTY50 or NIFTY10')


        # closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])

        # closing_prices_df = closing_prices_df[assets_input]
        
        # print(closing_prices_df.loc[0, stock])
        
        first_row_prices = closing_prices_df.iloc[0, 0:]
        
        df = closing_prices_df
        
        investment_values = first_row_prices * 100
        total_investment_amount = investment_values.sum()
        st.markdown("**Total Investment Amount (in rupees):**")
        st.write(np.round(total_investment_amount, 2))
        
        investment_per_stock = {stock: total_investment_amount * weight for stock, weight in weights_axis.items()}
        optimal_stocks_to_buy = {stock: investment // stock_prices_for_algo.loc[0, stock] for stock, investment in investment_per_stock.items()}
        optimal_stocks_at = {stock: investment // stock_prices_for_algo.loc[0, stock] for stock, investment in investment_per_stock.items()}
        #st.write(optimal_stocks_at)
        st.markdown("**Optimal Number of Stocks to buy (weights given in the attribution report):**")
        #st.write(optimal_stocks_to_buy)
        #st.text(f"{'Stock':<25}{'Stocks to buy':>15}")
        
        for stock, value in optimal_stocks_to_buy.items():
            st.text(f"{stock:<25}{value:>15}")


        portfolio_valuess = df.apply(lambda row: sum(
        row[stock] * optimal_stocks_to_buy[stock] for stock in optimal_stocks_to_buy), axis=1)
        df['Portfolio Value'] = portfolio_valuess
        st.markdown("**Portfolio Value of Attribution Report:**")
        st.dataframe(df.tail())

        # Reset the index if needed
        closing_prices_df.reset_index(inplace=True)

        df = closing_prices_df

        # Ensure 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(closing_prices_df['Date'])
        # print(df)  
        #st.dataframe(df)
        df['Date'] = pd.to_datetime(closing_prices_df['Date'])   # about this line as well.       
        print("printing df ", df)  

        
        fig = px.line(df, x='Date', y='Portfolio Value', 
                    title='Portfolio Value Over Time - Attribution Report',
                    labels={'Portfolio Value': 'Portfolio Value(AMAR)'},
                    markers=True, color_discrete_sequence=['red'],)
        
        fig.update_traces(name='Portfolio Value AMAR', showlegend=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (in rupees)', autosize=False, width=1000, height=600)
        st.plotly_chart(fig)

        print(df.columns)
        
        ## need to resolve this import too.  its not required to import files everytime. 
        investment_per_stock_us = {stock: total_investment_amount * weight for stock, weight in stock_dict.items()}
        optimal_stocks_to_buy_us = {stock: investment // df.loc[0, stock] for stock, investment in investment_per_stock_us.items()}
        optimal_stocks_qk = {stock: investment // df.loc[0, stock] for stock, investment in investment_per_stock_us.items()}
        #st.write(optimal_stocks_to_buy_us)
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

        # df['Date'] = pd.to_datetime(df['Date'])
        fig_port = px.line(df, x='Date', y='Portfolio Value', 
                    title='Portfolio Value Over Time - Qkrishi',
                    labels={'Portfolio Value': 'Portfolio Value(Qkrishi)'},
                    markers=True, color_discrete_sequence=['blue'])

        fig_port.update_traces(name='Portfolio Value Qkrishi', showlegend=True)
        fig_port.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (in rupees)', autosize=False, width=1000, height=600)
        st.plotly_chart(fig_port)

        print(portfolio_values)
        print(closing_prices_df['Date'])

        fig_compare = go.Figure()


        fig_compare.add_trace(go.Scatter(x= closing_prices_df['Date'], 
                    y=  portfolio_valuess,
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
        
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        def build_bqm(alpha, _mu, _sigma, cardinality):
            n = len(_mu)
            mdl = Model(name="stock_selection")
            x = mdl.binary_var_list(range(n), name="x")

            objective = alpha * (x @ _sigma @ x) - _mu @ x

            # cardinality constraint
            mdl.add_constraint(mdl.sum(x) == cardinality)
            mdl.minimize(objective)

            qp = from_docplex_mp(mdl)
            qubo = QuadraticProgramToQubo().convert(qp)

            bqm = dimod.as_bqm(
                qubo.objective.linear.to_array(),
                qubo.objective.quadratic.to_array(),
                dimod.BINARY,)
            return bqm
        
        #init_holding_amar = optimal_stocks_to_buy
        #st.text('init_holding_amar:')
        #st.write(init_holding_amar)

        def process_portfolio(init_holdings):
            cfg.hpfilter_lamb = 6.25
            cfg.q = 1.0  # risk-aversion factor
            # classical
            cfg.fmin = 0.01  # 0.001
            cfg.fmax = 0.5  # 0.5

            tickers = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 
                'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 
                'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
                'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 
                'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 
                'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 
                'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
            
            use_local = True
            if use_local is False:
                end_date = datetime.datetime(2024, 2, 26)
                start_date = datetime.datetime(2023, 2, 1)
                adj_close_df = pd.DataFrame()
                for ticker in tickers:
                    data = yf.download(ticker, start=start_date, end=end_date)
                    adj_close_df[ticker] = data['Adj Close']
                adj_close_df.to_csv('nifty50.csv')
            data = pd.read_csv('nifty50.csv', parse_dates=['Date'])

            constituents = pd.read_csv('constituents_nifty50.csv')
            sector_map = constituents.loc[constituents['symbol'].isin(tickers)]
            dates = data["Date"].to_numpy()
            monthly_df = data.resample('3M', on='Date').last() # resample to every 3 months
            month_end_dates = monthly_df.index
            available_sectors, counts = np.unique(np.array(sector_map.sector.tolist()), return_counts=True)

            total_budget = total_investment_amount
            num_months = len(month_end_dates)
            first_purchase = True 
            result = {}
            update_values = [0]
            months = []
            start_month = 0
            headers = ['Date', 'Value'] + list(tickers) + ['Risk', 'Returns', 'SR']
            opt_results_df = pd.DataFrame(columns=headers)
            row = []
            tickers = np.array(tickers)
            wallet = 0.0
            #portfolio_name = 'qkrishi'

            for i, end_date in enumerate(month_end_dates[start_month:]):
                df = data[dates <= end_date].copy()
                df.set_index('Date', inplace=True)
                months.append(df.last_valid_index().date())
                if first_purchase:
                    budget = total_budget
                    initial_budget = total_budget
                else:
                    value = sum([df.iloc[-1][s] * init_holdings.get(s, 0) for s in tickers]) # portfolio
                    #print(i, f"Portfolio : {budget:.2f},")
                    #print(f"Profit: {budget - initial_budget:.2f}")
                    update_values.append(budget - initial_budget)
                
                for s in df.columns:
                    cycle, trend = hpfilter(df[s], lamb=cfg.hpfilter_lamb)
                    df[s] = trend
                
                log_returns = np.log(df) - np.log(df.shift(1))
                null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
                drop_stocks = df.columns[null_indices]
                log_returns = log_returns.drop(columns=drop_stocks)
                log_returns = log_returns.dropna()
                tickers = log_returns.columns

                mu = log_returns.mean().to_numpy() * 252
                sigma = log_returns.cov().to_numpy() * 252
                price = df.iloc[-1] # last day price

                #Sell Idea
                threshold = 4 # Sell all stocks for `threshold` companies
                tickers_holding = np.array(list(init_holdings.keys())) # Names of the companies in initial holdings
                indices = np.in1d(tickers, tickers_holding) # Indices of `tickers_holding` in the list of all companies `tickers`
                argsort_indices = np.argsort(mu[indices]) # Obtain `mu` values at `indices`. Sort it.

                sell_indices =  argsort_indices < threshold # indices of the least `threshold` companies (least in terms of mu value)
                sell_tickers = tickers_holding[argsort_indices][sell_indices] # names of those companies

                # get the sectors of those companies. we will sell all stocks of them
                sectors = sector_map.loc[sector_map['symbol'].isin(sell_tickers)]['sector'].tolist()
                sectors = set(sectors) # remove duplicates

                tickers_new = sector_map.loc[sector_map['sector'].isin(sectors)]['symbol'].tolist()
                tickers_new = np.intersect1d(np.array(tickers_new), np.array(tickers))
                tickers_new = np.setdiff1d(np.array(tickers_new), np.array(sell_tickers))

                keep_indices = np.in1d(np.array(tickers), tickers_new)
                mu_new = mu[keep_indices]
                sigma_new = sigma[keep_indices][:, keep_indices]

                sales_revenue = 0.0
                for tick in sell_tickers:
                    sales_revenue += init_holdings[tick] * price[tick]
                    init_holdings.pop(tick, None) # remove that company from holdings

                bqm = build_bqm(cfg.q, mu_new, sigma_new, threshold)
                sampler_sa = SimulatedAnnealingSampler()
                result_sa = sampler_sa.sample(bqm, num_reads=5000)
                selection = list(result_sa.first.sample.values())
                selection = np.array(selection, dtype=bool)

                tickers_selected = tickers_new[selection]

                keep_indices = np.in1d(tickers_new, tickers_selected)
                mu_selected = mu_new[keep_indices]
                sigma_selected = sigma_new[keep_indices][:, keep_indices]

                qpo = SinglePeriod(cfg.q, 
                    mu_selected, 
                    sigma_selected, 
                    sales_revenue + wallet, 
                    np.array([price[tick] for tick in tickers_selected]), 
                    tickers_selected)
                solution = qpo.solve_cqm(init_holdings)
                result = solution['stocks'].copy()
                asset_weights = qpo._weight_allocation(solution['stocks'])
                optimal_weights_dict = qpo._get_optimal_weights_dict(
                asset_weights, solution['stocks'])

                metrics = qpo._get_risk_ret(asset_weights) # risk, returns and sharpe ratio

                for tick, val in result.items():
                    if val != 0:
                        #print(f"{tick}, ({sector_map.loc[sector_map['symbol'] == tick]['sector'].tolist()[0]})", ' '*2, val)
                        if tick not in init_holdings.keys():
                            init_holdings[tick] = val
                        else:
                            init_holdings[tick] += val
                value = sum([price[s] * result.get(s, 0.0) for s in tickers_new]) # Amount invested in purchasing
                value_port = sum([price[s] * init_holdings.get(s, 0.0) for s in init_holdings]) # Portfolio Value after Rebalancing
                wallet = (sales_revenue + wallet) - value # Amount left in wallet

                returns = f"{metrics['returns']:.2f}"
                risk = f"{metrics['risk']:.2f}"
                sr = f"{metrics['sharpe_ratio']:.2f}"

                row = [months[-1].strftime('%Y-%m-%d'), value_port/initial_budget] + \
                    [init_holdings.get(s, 0) for s in tickers] + \
                    [risk, returns, sr] 
                
                opt_results_df.loc[i] = row.copy()
                first_purchase = False
            return opt_results_df
        
        #init_holding_amar = optimal_stocks_to_buy
        #st.write(init_holding_amar)
        #init_holding_qkrishi = optimal_stocks_to_buy_us


        # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @  @ @ @ @ @ @ @ @  @ @ @ @ @ @ @  @ @ @ @ @@ @ @ @  @ @ @
        #st.text('optimal_stocks_to_buy:')
        #st.write(optimal_stocks_to_buy)
        process_portfolio_amar = process_portfolio(optimal_stocks_to_buy)
        #st.write(process_portfolio_amar)
        process_portfolio_amar_df = process_portfolio_amar.to_csv('rebalancing_amar.csv')
        dataf = pd.read_csv('rebalancing_amar.csv')
        #st.write(optimal_stocks_at)
        new_data_dict = optimal_stocks_at
        #st.text('new_data_dict:')
        #st.write(optimal_stocks_at)
        new_row_df = pd.DataFrame(new_data_dict, index=[0])
        for column in dataf.columns:
            if column not in new_row_df.columns:
                new_row_df[column] = "" if column == "Date" else 0

        new_row_df = new_row_df[dataf.columns]
        updated_dataf = pd.concat([new_row_df, dataf], ignore_index=True)

        tickerss = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 
                            'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 
                            'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
                            'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 
                            'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 
                            'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 
                            'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
        
        for column in updated_dataf.columns:
            if column in tickerss:
                updated_dataf[column] = updated_dataf[column].diff().fillna(updated_dataf[column])
        updated_dataf = updated_dataf.drop('Unnamed: 0', axis=1)
        columns_to_style = tickerss
        def apply_styling(value):
                if value > 0:
                    return f'<span style="color: green;">{value}</span>'
                elif value < 0:
                    return f'<span style="color: red;">{value}</span>'
                else:
                    return value
        for column in columns_to_style:
                updated_dataf[column] = updated_dataf[column].apply(apply_styling)

        updated_dataf["Value"] = updated_dataf["Value"] * total_investment_amount
        updated_dataf = updated_dataf.loc[:, (updated_dataf != 0).any(axis=0)] # to remove columns with 0s
        updated_dataf = updated_dataf.to_html(float_format=lambda x: '{:.2f}'.format(x), escape=False)
        #st.write(new_data_dict)
        st.write("AMAR's Portfolio after Rebalancing:")
        st.write(updated_dataf, unsafe_allow_html=True)
        df_fta = pd.read_html(updated_dataf)[0]
        df_fta['Date'] = pd.to_datetime(df_fta['Date'])
        # @ @ @@ @ @ @ @ @ @ @ @ @ @ @  @ @ @ @ @ @ @ @ @ @ @ @ @ @ @  @ @ @  @ @  @ @ @ @ @ @ 

        process_portfolio_qkrishi = process_portfolio(optimal_stocks_to_buy_us)
        #st.write(process_portfolio_amar)
        process_portfolio_qkrishi_df = process_portfolio_qkrishi.to_csv('rebalancing_qkrishi.csv')
        datafq = pd.read_csv('rebalancing_qkrishi.csv')
        new_data_dictq = optimal_stocks_qk
        #st.write(new_data_dictq)
        new_row_dfq = pd.DataFrame(new_data_dictq, index=[0])
        for column in datafq.columns:
            if column not in new_row_dfq.columns:
                new_row_dfq[column] = "" if column == "Date" else 0

        new_row_dfq = new_row_dfq[datafq.columns]
        updated_datafq = pd.concat([new_row_dfq, datafq], ignore_index=True)

        tickersss = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 
                            'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 
                            'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
                            'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 
                            'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 
                            'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 
                            'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
        
        for column in updated_datafq.columns:
            if column in tickersss:
                updated_datafq[column] = updated_datafq[column].diff().fillna(updated_datafq[column])
        updated_datafq = updated_datafq.drop('Unnamed: 0', axis=1)
        columns_to_style = tickersss
        def apply_styling(value):
                if value > 0:
                    return f'<span style="color: green;">{value}</span>'
                elif value < 0:
                    return f'<span style="color: red;">{value}</span>'
                else:
                    return value
        for column in columns_to_style:
                updated_datafq[column] = updated_datafq[column].apply(apply_styling)

        updated_datafq["Value"] = updated_datafq["Value"] * total_investment_amount
        updated_datafq = updated_datafq.loc[:, (updated_datafq != 0).any(axis=0)] # to remove all columns with 0s
        updated_datafq = updated_datafq.to_html(float_format=lambda x: '{:.2f}'.format(x), escape=False)
        #st.write(new_data_dict)
        st.write("QKRISHI's Portfolio after Rebalancing:")
        st.write(updated_datafq, unsafe_allow_html=True)
        df_ftq = pd.read_html(updated_datafq)[0]
        df_ftq['Date'] = pd.to_datetime(df_fta['Date'])
        

        fig_rebalancing = go.Figure()
        fig_rebalancing.add_trace(go.Scatter(x=df_fta['Date'], y=df_fta['Value'], mode='lines+markers', name='Bank Portfolio+ Qkrishi Rebalancing', line=dict(color='red'), showlegend=True))
        fig_rebalancing.add_trace(go.Scatter(x=df_ftq['Date'], y=df_ftq['Value'], mode='lines+markers', name='Qkrishi Portfolio  + Qkrishi Rebalancing', line=dict(color='blue'), showlegend=True))
        
        fig_rebalancing.update_layout(title='Rebalanced Portfolio Values Over Time',
            xaxis_title='Date', 
            yaxis_title='Portfolio Value',
            autosize=False, 
            width=1000, 
            height=600,
            yaxis_range=[1500000,2200000])
        
        st.plotly_chart(fig_rebalancing)
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # # # 

    else:
        st.write("The number of assets entered cannot be compared with the Attribution Report. But let us build a portfolio based on the given assets.")
        tickers = assets_input.replace(' ', '').split(',')  # Remove spaces and split by comma
        if tickers and tickers[0]:  # Check if there's at least one ticker
            # Initialize an empty DataFrame to hold closing prices
            close_prices = pd.DataFrame()

            for ticker in tickers:
            # Downloading the stock data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
                if not data.empty:
                    # Extracting the closing prices and adding them to the DataFrame
                    close_prices[ticker] = data['Close']
                else:
                    st.write(f"No data found for {ticker} with the selected date range.")

            if not close_prices.empty:
            # Displaying the closing prices
                closing_prices_df = pd.DataFrame(close_prices)
                closing_prices_df.to_csv('stock_closing_prices.csv')
                stock_closing_prices = pd.read_csv('stock_closing_prices.csv')
                closing_prices_reset = close_prices.reset_index(drop=True)
                st.markdown("**Closing Prices:**")
                st.dataframe(stock_closing_prices)

                class cfg:
                    hpfilter_lamb = 6.25
                    q = 1.0
                    fmin = 0.001
                    fmax = 0.5
                    num_stocks = len(closing_prices_reset.columns)
            
                for s in closing_prices_reset.columns:
                    cycle, trend = hpfilter(closing_prices_reset[s], lamb=cfg.hpfilter_lamb)
                    closing_prices_reset[s] = trend

                log_returns = np.log(closing_prices_reset) - np.log(closing_prices_reset.shift(1))
                null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
                drop_stocks = closing_prices_reset.columns[null_indices]
                log_returns = log_returns.drop(columns=drop_stocks)
                log_returns = log_returns.dropna()
                tickers = log_returns.columns
                cfg.num_stocks = len(tickers)
                mu = log_returns.mean().to_numpy() * 252
                sigma = log_returns.cov().to_numpy() * 252

                plt.figure(figsize = (4,4))
                for s in log_returns.columns:
                    plt.plot(closing_prices_reset[s], label=s)
                legend_fontsize = 8
                plt.legend(loc="upper center", bbox_to_anchor=(2.0, 1.1), fancybox=True, shadow=True, ncol=4, fontsize=legend_fontsize)
                plt.xlabel("Days")
                plt.ylabel("Stock Prices")
                plt.title("Stock Prices Over Time")
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()

                cfg.kappa = cfg.num_stocks
            
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

                study_mvo_miqp.optimize(lambda trial: objective_mvo_miqp(trial, mu, sigma), n_trials=25-len(study_mvo_miqp.trials), n_jobs=1)
                trial_mvo_miqp = study_mvo_miqp.best_trial
                cpo = ClassicalPO(mu, sigma, cfg)
                cpo.cfg.gamma = 1.9937858736079478
                res = cpo.mvo_miqp()
                weights = res['w']
                stock_dict = dict(zip(tickers, np.around(weights, 5)))
                stock_dict = {i: j for i, j in stock_dict.items() if j != 0}

            
                st.markdown("**Weights Allocated by the Algorithm:**")
                #st.text(f"{'Stock':<25}{'Weights(%)':>15}")
                for stock, value in stock_dict.items():
                    percentage = round(value * 100, 2)  # Multiply by 100 and round off to 2 decimal places
                    st.text(f"{stock:<25}{percentage:>15}")
                    
                st.markdown("**Returns and Risk of the portfolio:**")
                mvo_miqp_res = cpo.get_metrics(weights)
                for metric, value in mvo_miqp_res.items():
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

                first_row_prices = close_prices.iloc[0, 0:]
                investment_values = first_row_prices * 100
                total_investment_amount = investment_values.sum()
                st.markdown("**Total Investment Amount (in rupees):**")
                st.write(total_investment_amount)

                #df = pd.read_csv('stock_closing_prices.csv')
                investment_per_stock_us = {stock: total_investment_amount * weight for stock, weight in stock_dict.items()}
                optimal_stocks_to_buy_us = {stock: investment // stock_closing_prices.loc[0, stock] for stock, investment in investment_per_stock_us.items()}
                st.markdown("**Optimal Number Stocks to buy (weights given by the Algorithm):**")
                #st.write(optimal_stocks_to_buy_us)
                #st.text(f"{'Stock':<25}{'Stocks to buy':>15}")
                for stock, value in optimal_stocks_to_buy_us.items():
                    st.text(f"{stock:<25}{value:>15}")

                portfolio_values = stock_closing_prices.apply(lambda row: sum(
                row[stock] * optimal_stocks_to_buy_us[stock] for stock in optimal_stocks_to_buy_us), axis=1)
                stock_closing_prices['Portfolio Value'] = portfolio_values
                st.markdown("**Portfolio Value given by the Algorithm:**")
                st.dataframe(stock_closing_prices.tail())

                stock_closing_prices['Date'] = pd.to_datetime(stock_closing_prices['Date'])
                fig_port = px.line(stock_closing_prices, x='Date', y='Portfolio Value', 
                            title='Portfolio Value Over Time',
                            labels={'Portfolio Value': 'Portfolio Value'},
                            markers=True, color_discrete_sequence=['blue'])
            
                fig_port.update_traces(name='Portfolio Value', showlegend=True)
                fig_port.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (in rupees)', autosize=False, width=1000, height=600)
                st.plotly_chart(fig_port)
