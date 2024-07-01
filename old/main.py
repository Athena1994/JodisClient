import numpy as np
import pandas as pd

from ai.training_manager import TrainingManager
from data.loader.sql_loader import SQLOHCLLoader

'''
Data: 
    Data is provided by using a OHCLLoader that produces Open-High-Close-Low 
    candle data wrapped in a pandas DataFrame

training environment:
    The entire training is wrapped inside TrainingManager. 
  
    TrainingManager:
        Data is split into 3 sets, each consisting of (precursor, data, epilog):
            - training 
            - validation
            - test

        On a run, every training chunk is iteratively set onto an Environment- 
        Simulation with a reset wallet. For every sample in every chunk 
        notify_update() is called on the sim.
                
    TradingEnvironment:
        A TradingEnvironment provides access to the current wallet and 
        offers a buy and sell method.
    
    EnvironmentSimulation
        Implements a TradingEnvironment.
        
        On each tick, the following happens:
            1. action selection
            2. reward calculation
            3. transitioning to new state
            4. add experience tuple to replay buffer
    
        notify_update() triggers a TradingInterface update which in turns calls
        _tick()
    
    Agents:
        Selects an action (hold, buy, sell) based on current state and 
        data obtained from a TradingInterface.
        
        BaseIndicatorAgent:
            Provides a list of indicators this agent requires.
            Calls a DecisionArbiter for selecting an action.
            
        SimpleIndicatorAgent(BaseIndicatorAgent):
            Uses all indicators available.
            
    DecisionArbiter:
        Decides on an action based on a feature_vec (DataFrame)
        
        RandomArbiter:
            self explanatory
            
        QArbiter:
            Uses a QFunction to assign q-values to all possible actions for a 
            given state/feature combination
        
    QFunction:
        Maps a feature and state vector to action q values.
        
        DeepQFunction:
            Uses a QNN network for mapping.
        
    ReplayBuffer:
        The ReplayBuffer memorizes every training-sample/state tuple
    
'''


SQL_USER = 'trader'
SQL_PW = 'trade_good'
SQL_SERVER = 'vserver'
SQL_DATABASE = 'trading_data'
SQL_TABLE = 'ohcl'


loader = SQLOHCLLoader(SQL_USER, SQL_PW, SQL_SERVER, SQL_DATABASE, SQL_TABLE)

data_df = loader.get('BTCEUR', 'ONE_MINUTE')

sim = TrainingManager(data_df)
sim.run()

