import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import json
matplotlib.use('Agg')
# %matplotlib inline
from datetime import datetime, timedelta
import StockPortfolioEnv as SP_Env

from finrl.apps import config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

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

class RL_Agent:
    def __init__(self, portfolio_name, portfolio_stocks, portfolio_principal):
        self.portfolio_name = portfolio_name
        self.portfolio_stocks = portfolio_stocks
        self.portfolio_principal = portfolio_principal
    
    import sys
    sys.path.append("../FinRL-Library")
    def train_portfolio(self):
        try:
            import os
            if not os.path.exists("./" + config.DATA_SAVE_DIR):
                os.makedirs("./" + config.DATA_SAVE_DIR)
            if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
                os.makedirs("./" + config.TRAINED_MODEL_DIR)
            if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
                os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
            if not os.path.exists("./" + config.RESULTS_DIR):
                os.makedirs("./" + config.RESULTS_DIR)
            today = datetime.today().strftime('%Y-%m-%d')
            date_splits = get_date_split(self.portfolio_stocks, 0.7)

            df = YahooDownloader(start_date = date_splits['train_start'],
                                end_date = today,
                                ticker_list = self.portfolio_stocks).fetch_data()
            fe = FeatureEngineer(
                                use_technical_indicator=True,
                                use_turbulence=False,
                                user_defined_feature = False)

            df = fe.preprocess_data(df)
            df=df.sort_values(['date','tic'],ignore_index=True)
            df.index = df.date.factorize()[0]

            cov_list = []
            return_list = []

            # look back is one year
            lookback=252
            for i in range(lookback,len(df.index.unique())):
                data_lookback = df.loc[i-lookback:i,:]
                price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
                return_lookback = price_lookback.pct_change().dropna()
                return_list.append(return_lookback)
                covs = return_lookback.cov().values 
                cov_list.append(covs)

            
            df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
            df = df.merge(df_cov, on='date')
            df = df.sort_values(['date','tic']).reset_index(drop=True)

            train = data_split(df, date_splits['train_start'],date_splits["train_stop"])

            stock_dimension = len(train.tic.unique())
            state_space = stock_dimension
            print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

            env_kwargs = {
                "hmax": 100, 
                "initial_amount": self.portfolio_principal, 
                "transaction_cost_pct": 0.05, 
                "state_space": state_space, 
                "stock_dim": stock_dimension, 
                "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
                "action_space": stock_dimension, 
                "reward_scaling": 1e-1
                
            }

            e_train_gym = SP_Env.StockPortfolioEnv(df = train, **env_kwargs)

            env_train, _ = e_train_gym.get_sb_env()

            agent = DRLAgent(env = env_train)
            PPO_PARAMS = {
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.001,
                "batch_size": 128,
            }

            model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
            trained_ppo = agent.train_model(model=model_ppo,model_name=f"trained_PPO_{self.portfolio_name}", tb_log_name='ppo', total_timesteps=50000)
            return {
                "status":200
            }
        except Exception as e:
            print(e)
            return {
                "status":500
            }



    def get_portfolio_prediction(self):
        import os
        cwd = os.getcwd()
        trained_model_name = ("trained_PPO_"+self.portfolio_name).upper()
        trained_model_path = cwd+f"\\trained_models\\{trained_model_name}.zip"
        print(trained_model_path)
        
        today = datetime.today().strftime('%Y-%m-%d')
        date_splits = get_date_split(self.portfolio_stocks, 0.7)
        df = YahooDownloader(start_date = date_splits['train_start'],
                                end_date = today,
                                ticker_list = self.portfolio_stocks).fetch_data()
        fe = FeatureEngineer(
                            use_technical_indicator=True,
                            use_turbulence=False,
                            user_defined_feature = False)

        df = fe.preprocess_data(df)
        df=df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []
        return_list = []

        # look back is one year
        lookback=252
        for i in range(lookback,len(df.index.unique())):
            data_lookback = df.loc[i-lookback:i,:]
            price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
            covs = return_lookback.cov().values 
            cov_list.append(covs)

        
        df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)

        trade = data_split(df,date_splits['trade_start'], today)
        stock_dimension = len(trade.tic.unique())
        state_space = stock_dimension
        
        env_kwargs = {
                "hmax": 100, 
                "initial_amount": self.portfolio_principal, 
                "transaction_cost_pct": 0.05, 
                "state_space": state_space, 
                "stock_dim": stock_dimension, 
                "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
                "action_space": stock_dimension, 
                "reward_scaling": 1e-1
                
            }
        e_trade_gym = SP_Env.StockPortfolioEnv(df = trade, **env_kwargs)
        from pyfolio import timeseries

        df_daily_return_ppo, df_actions_ppo = DRL_prediction_load("ppo",e_trade_gym,trained_model_path)
        # ppo_cumpod =(df_daily_return_ppo.daily_return+1).cumprod()-1
        

        DRL_strat_ppo = convert_daily_return_to_pyfolio_ts(df_daily_return_ppo)
        perf_func = timeseries.perf_stats 
        ppo_cumpod = (DRL_strat_ppo+1).cumprod()-1
        ppo_cumpod = pd.Series(data=ppo_cumpod.values, index=map(lambda x : x.to_pydatetime().strftime("%Y-%m-%d"), ppo_cumpod.index))
        ppo_cumpod.index = pd.to_datetime(ppo_cumpod.index)
        ppo_cumpod = ppo_cumpod.resample('M').mean()
        ppo_cumpod.columns = ["monthly_return"]
        ppo_cumpod = ppo_cumpod.to_json(orient="index")
        perf_stats_all_ppo = perf_func( returns=DRL_strat_ppo,factor_returns=DRL_strat_ppo, positions=None, transactions=None, turnover_denom="AGB")
        latest_action = df_actions_ppo.tail(1).to_json(orient="records")
        suggested_allocation = []
        serialized_actions = json.loads(latest_action)
        sorted_actions = {k:v for k,v in sorted(serialized_actions[0].items(), key=lambda item: item[1], reverse=True)}
        for k,v in serialized_actions[0].items():
            v = round(v,4)
            suggested_allocation.append({
                "ticker": k,
                "weight": v,
            })
        model_stats = perf_stats_all_ppo.to_dict()
        # model_stats['Suggested_allocation'] = suggested_allocation
        model_stats['Suggested Allocation'] = [sorted_actions]
        model_stats['Cumulative Mean Monthly Return'] = json.loads(ppo_cumpod)
        return model_stats