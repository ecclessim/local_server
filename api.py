from flask import Flask, request
from pandas_datareader import data as web
import numpy as np
import yfinance as yf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns, EfficientSemivariance
from pypfopt import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions
from werkzeug.serving import WSGIRequestHandler
import json

app = Flask(__name__)

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
def DRL_prediction_load(model_name, environment, cwd):
    MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
    if model_name not in MODELS:
        raise NotImplementedError("NotImplementedError")
    try:
        # load agent
        model = MODELS[model_name].load(cwd)
        print("Successfully load model", cwd)
    except BaseException:
        raise ValueError("Fail to load agent!")
    test_env, test_obs = environment.get_sb_env()
    """make a prediction"""
    account_memory = []
    actions_memory = []
    test_env.reset()
    for i in range(len(environment.df.index.unique())):
        action, _states = model.predict(test_obs)
        # account_memory = test_env.env_method(method_name="save_asset_memory")
        # actions_memory = test_env.env_method(method_name="save_action_memory")
        test_obs, rewards, dones, info = test_env.step(action)
        if i == (len(environment.df.index.unique()) - 2):
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("hit end!")
            break
    return account_memory[0], actions_memory[0]

def get_date_split(stocks, train):
    today = datetime.today()
    first_data = yf.Ticker(stocks[0]).history(period ="max")
    first_record = first_data.head(1).index.values
    for ticker in stocks:
        data = yf.Ticker(ticker).history(period="max")
        if first_record < data.head(1).index.values:
            first_record = data.head(1).index.values 
    train_start_date = datetime.utcfromtimestamp(first_record[0].tolist()/1e9)
    delta = today - train_start_date
    train_days = int(delta.days * train)
    trade_days = int(delta.days * (1-train))
    print(f"Training days: {train_days}, trade days: {trade_days}")
    train_stop_date = train_start_date + timedelta(days = train_days)
    trade_start_date = train_stop_date + timedelta(days = 1)
    trade_stop_date = today
    return {
        "train_start": train_start_date.strftime('%Y-%m-%d'),
        "train_stop": train_stop_date.strftime('%Y-%m-%d'),
        "trade_start": trade_start_date.strftime('%Y-%m-%d'),
        "trade_stop": trade_stop_date.strftime('%Y-%m-%d'),
    }
def gen_portfolio_series(input_quotes, start_date):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(input_quotes,start=start_date, end=today)
    portfolio_close = data.loc[:,"Adj Close"]
    return portfolio_close

def string_percent(value, degree):
    converted = str(round(value, degree) * 100) + '%'
    return converted

def existing_port_weight_gen(asset_values, principal):
    weights = []
    for asset in asset_values:
        percentage = asset/principal
        weights.append(percentage)
    return weights

def getAllocation(df,weight,portf_value):
    latest_prices = get_latest_prices(df)
    weights_optimised = weight
    da = DiscreteAllocation(weights_optimised, latest_prices, total_portfolio_value=portf_value)
    allocation, leftover = da.lp_portfolio()
    return [allocation, leftover]
    
# INPUTS FOR CALCULATE RETURNS: ASSETS, WEIGHTS, START_DATE
@app.route('/calculate_returns', methods=['GET'])
def calculate_returns():
    principal = float(request.args.get('principal'))
    assets = request.args.get('assets')
    weights = request.args.get('weights')
    weights = [float(x) for x in weights.split()] 
    weights = np.array(weights)
    start_date = request.args.get('start_date')
    df = gen_portfolio_series(assets,start_date)
    # time series for closing prices 
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual,weights))
    port_volatility = np.sqrt(port_variance)
    port_simple_annual_return = np.sum(returns.mean() * weights) * 252
    returns['pf_daily_returns'] = returns.dot(weights)
    returns['pf_cumulative_returns'] = (1 + returns['pf_daily_returns']).cumprod()
    port_cumulative_return = returns['pf_cumulative_returns'].iloc[-1]
    percent_var = port_variance
    percent_vol = port_volatility
    percent_ret = port_simple_annual_return
    dollar_return_simple = principal * port_simple_annual_return
    return {
        "simple annual return" : percent_ret,
        "cumulative return" : port_cumulative_return,
        "portfolio variance" : percent_var,
        "portfolio volatility" : percent_vol,
        "simple dollar return" : dollar_return_simple
    }
# INPUTS FOR SHARPE AND SORTINO: ASSETS, START_DATE, PRINCIPAL 
@app.route("/calculate_sharpe", methods=["GET"])
def sharpe_allocation():
    # mu = expected_returns.mean_historical_return(df)
    # EMA gives more weight to current data
    assets = request.args.get('assets')
    start_date = request.args.get('start_date')
    principal = float(request.args.get('principal'))
    df = gen_portfolio_series(assets,start_date)
    mu = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu,S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    verbose_sharpe_weights = ef.max_sharpe()
    verbose_sharpe_weights = ef.clean_weights()
    sharpe_perf = ef.portfolio_performance()
    exp_return_sharpe = sharpe_perf[0]
    exp_vlt_sharpe = sharpe_perf[1]
    # exp_dollar_return_sharpe = "{:,}".format(round(principal * exp_return_sharpe,3))
    exp_dollar_return_sharpe = round(principal*exp_return_sharpe,2)
    allocation = getAllocation(df,verbose_sharpe_weights,principal)
    verbose_allocation = allocation[0]
    verbose_allocation = dict((k,int(v)) for k,v in verbose_allocation.items())
    suggested_allocation = []
    sharpe_weights = []
    sw_tickers = list(verbose_sharpe_weights.keys())
    sw_values = list(verbose_sharpe_weights.values())
    for i, val in enumerate(sw_tickers):
        sharpe_weights.append(
            {
                "ticker": val,
                 "weight": sw_values[i]
            }
        )
    for k,v in verbose_allocation.items():
        suggested_allocation.append({
            "ticker":k,
            "shares":int(v)
        })
    return {

        "Expected Annual Return" : round(exp_return_sharpe,2), 
        "Expected Annual Volatility" : round(exp_vlt_sharpe,2),
        "Expected Dollar Return" : exp_dollar_return_sharpe,
        "Sharpe Ratio" : round(sharpe_perf[2],2),
        "Sharpe Weights" : sorted(sharpe_weights, key=lambda k: k['weight'], reverse=True),
        "Suggested Allocation" : suggested_allocation,
        "Leftover($)" :  allocation[1]
    }

@app.route("/calculate_sortino", methods=["GET"])
def sortino_allocation():
    assets = request.args.get('assets')
    start_date = request.args.get('start_date')
    principal = float(request.args.get('principal'))
    df = gen_portfolio_series(assets,start_date)
    historical_returns = expected_returns.returns_from_prices(df)
    mu = expected_returns.ema_historical_return(df)
    es = EfficientSemivariance(mu, historical_returns)
    es.efficient_return(0.05)
    # We can use the same helper methods as before
    verbose_sortino_weights = es.clean_weights()
    sortino_perf = es.portfolio_performance()
    exp_return_sortino = sortino_perf[0]
    exp_vlt_sortino = sortino_perf[1]
    # commafy dollar return
    # expected_dollar_return_sortino = "{:,}".format(round(principal * exp_return_sortino,3))
    expected_dollar_return_sortino = round(principal*exp_return_sortino,2)
    allocation = getAllocation(df,verbose_sortino_weights,principal)
    verbose_allocation = allocation[0]
    suggested_allocation = []
    sw_tickers = list(verbose_sortino_weights.keys())
    sw_values = list(verbose_sortino_weights.values())
    sortino_weights = []
    for i, val in enumerate(sw_tickers):
        sortino_weights.append(
            {
                "ticker": val,
                 "weight": sw_values[i]
            }
        )
    for k,v in verbose_allocation.items():
        suggested_allocation.append({
            "ticker":k,
            "shares":int(v)
        })
    return {

        "Expected Annual Return" : round(exp_return_sortino,2), 
        "Expected Annual Volatility" : round(exp_vlt_sortino,2),
        "Expected Dollar Return" : round(expected_dollar_return_sortino,2),
        "Sortino Ratio" : round(sortino_perf[2],2),
        "Sortino Weights" : sorted(sortino_weights, key=lambda k: k['weight'], reverse=True),
        "Suggested Allocation" : suggested_allocation,
        "Leftover($)" :  round(allocation[1],2)
    }

@app.route("/calculate_min_volatility", methods=["GET"])
def min_volatility_allocation():
    # mu = expected_returns.mean_historical_return(df)
    # EMA gives more weight to current data
    assets = request.args.get('assets')
    start_date = request.args.get('start_date')
    principal = float(request.args.get('principal'))
    df = gen_portfolio_series(assets,start_date)
    mu = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu,S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    verbose_min_vol_weights = ef.min_volatility()
    verbose_min_vol_weights = ef.clean_weights()
    min_vol_perf = ef.portfolio_performance()
    exp_return_min_vol = min_vol_perf[0]
    exp_vlt_min_vol = min_vol_perf[1]
    exp_dollar_return_min_vol = round(principal * exp_return_min_vol,2)
    allocation = getAllocation(df,verbose_min_vol_weights,principal)
    verbose_allocation = allocation[0]
    verbose_allocation = dict((k,int(v)) for k,v in verbose_allocation.items())
    mv_tickers = list(verbose_min_vol_weights.keys())
    mv_values = list(verbose_min_vol_weights.values())
    mv_weights = []
    suggested_allocation = []
    for i, val in enumerate(mv_tickers):
        mv_weights.append(
            {
                "ticker": val,
                 "weight": mv_values[i]
            }
        )
    for k,v in verbose_allocation.items():
        suggested_allocation.append({
            "ticker":k,
            "shares":int(v)
        })
    return {

        "Expected Annual Return" : round(exp_return_min_vol,2), 
        "Expected Annual Volatility" : round(exp_vlt_min_vol,2),
        "Expected Dollar Return" : round(exp_dollar_return_min_vol,2),
        "Sharpe Ratio" : round(min_vol_perf[2],2),
        "Minimum Volatility Weights" : sorted(mv_weights, key=lambda k: k['weight'], reverse=True),
        "Suggested Allocation" : suggested_allocation,
        "Leftover($)" :  round(allocation[1],2)
    }
@app.route("/get_company_info", methods=["GET"])
def get_company_info():
    ticker = request.args.get('ticker')
    try:
        company_info = yf.Ticker(ticker).info
        current_price = 0.0 
        if 'currentPrice' in company_info:
            print("Retrieving currentPrice from JSON response...")
            current_price = float(company_info.get('currentPrice', 0.0))
        else:
            print("Retrieving regularMarketPrice from JSON response...")
            if 'regularMarketPrice' in company_info and company_info.get('regularMarketPrice') is not None:
                current_price = company_info.get('regularMarketPrice', 0.0)
        response = {
            "logo_url" : company_info.get('logo_url', ''),
            "companyName" : company_info.get('longName', '') if 'longName' in company_info else company_info.get('shortName', ''),
            "ticker" : company_info.get('symbol', ''),
            "country" : company_info.get('country', ''),
            "sector" : company_info.get('sector', ''),
            "longSummary" : company_info.get('longBusinessSummary', '') if 'longBusinessSummary' in company_info else company_info.get('description', ''),
            "currentPrice" : current_price,
            "open" : float(company_info.get('open', 0.0)),
            "previousClose" : float(company_info.get('previousClose', 0.0)),
            "52WeekChange": float(company_info.get('52WeekChange', 0.0)) if company_info.get('52WeekChange') != None else 0.0,
            "volume" : float(company_info.get('volume', 0.0)) if company_info.get('volume', 0.0) != None else 0.0,
            "avgVolume" : int(company_info.get('averageVolume', 0)) if company_info.get('averageVolume') != None else 0,
            "mktCap" : int(company_info.get('marketCap', 0)) if company_info.get('marketCap') != None else 0,
            "sharesOutst" : int(company_info.get('sharesOutstanding', 0)),
            "forwardPE" : float(company_info.get('forwardPE')) if company_info.get('forwardPE') != None else 0.0,
            "divYield" : company_info.get('dividendYield', 0.0) if company_info.get('dividendYield') != None else 0.0,
            "yield" : company_info.get('yield', 0.0) if company_info.get('yield') != None else 0.0,
        }
        return response
    except Exception as e:
        print(f"Failed to fetch data: {e}")


@app.route("/get_time_series", methods=["GET"])
def get_time_series():
    ticker = request.args.get("ticker")
    period = request.args.get("period")
    interval = request.args.get('interval')
    data = yf.download(tickers=ticker, period=period,interval=interval,threads=True)
    if data is not None and len(data) > 0:
        data.reset_index(inplace=True)
        print(len(data))
        return data.to_json(orient='records', indent=2,date_format='iso') 
    else:
        return {
            "Error": "Unable to fetch data"
        }

@app.route("/get_stock_price", methods=["GET"])
def get_stock_price():
    tickers = request.args.get("tickers")
    price_list = []
    for ticker in list(tickers.split(" ")):
        data = yf.Ticker(ticker).history()
        previous_close = round(data.tail(2)['Close'][0],2)
        latest_quote = round(data.tail(2)['Close'][1],2)
        daily_change = round(latest_quote - previous_close,2)
        daily_change_pct = round(((latest_quote - previous_close)/previous_close)*100,2)
        price_list.append({'ticker': ticker, 'last_quote': latest_quote,'previous_close': previous_close, 'daily_change': daily_change, 'daily_change_pct': daily_change_pct})
    return json.dumps(price_list,indent=2)


import RL_Model
@app.route("/ppo_portfolio_analysis", methods=['GET'])
def ppo_portfolio_analysis():
    portfolio_name = request.args.get('portfolio_name')
    portfolio_stocks = request.args.get('portfolio_stocks')
    portfolio_principal = request.args.get('portfolio_principal')
    portfolio_stocks = list(portfolio_stocks.split(" "))
    portfolio_principal = float(portfolio_principal)
    trained_model_path = os.getcwd()+f"\\trained_models\\trained_PPO_{portfolio_name}.zip".upper()
    print(trained_model_path)
    rl_model = RL_Model.RL_Agent(portfolio_name,portfolio_stocks,portfolio_principal)
    # Check if trained model exists and created date is not > 4 months ago
    if os.path.exists(trained_model_path) and (datetime.now() - datetime.fromtimestamp(os.path.getctime(trained_model_path))).days < 90:
        print(f"Found model {trained_model_path}, loading model from disk")
        model_stats = rl_model.get_portfolio_prediction()
        return model_stats
    else: 
        print(f"No model found {trained_model_path}, training new model...")
        rl_model.train_portfolio()
        model_stats = rl_model.get_portfolio_prediction()
        return model_stats

@app.route("/train_ppo_model", methods=['GET'])
def train_ppo_model():
    portfolio_name = request.args.get('portfolio_name')
    portfolio_stocks = request.args.get('portfolio_stocks')
    portfolio_principal = request.args.get('portfolio_principal')
    portfolio_stocks = list(portfolio_stocks.split(" "))
    portfolio_principal = float(portfolio_principal)
    # trained_model_path = os.getcwd()+f"\\trained_models\\trained_PPO_{portfolio_name}.zip".upper()
    try:
        rl_model = RL_Model.RL_Agent(portfolio_name,portfolio_stocks,portfolio_principal)
        rl_model.train_portfolio()
        return {
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "failed: "+str(e),
        }


if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(debug=True,host='0.0.0.0', port=5000)
    # app.run(debug=True,port=5000)   

