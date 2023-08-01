import pandas as pd
from calendar_data import getCalendarData
from weather import getWeatherData
from entsoe_data import getEntsoeData
from data import getData, createLags, split_in_time, printMetrics
from tft_functions import tft_predict, printMetricsHourly

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree

plt.style.use('fivethirtyeight')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error


from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import copy
from pathlib import Path
import warnings


import lightning.pytorch as pl # Instead of import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torchmetrics import MeanAbsoluteError

import warnings
warnings.filterwarnings("ignore")
from tqdm.autonotebook import tqdm

#######################################################################################################################
# Program Functionality parameters

data_load = 1                                       # if want to load data or get it from csv         
search_hyperparameters = 0                          # if want to search for optimal tft hyperparameters
train = 0                                           # if want to train model or load from checkpoint
#######################################################################################################################
# Create dataset

fromDate = 20210101                                 # start date for data load
toDate = 20230701                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only CZ, HU or SK)
load_lag = [24, 48, 168]                            # lags we want to create

X = getData(data_load, location, fromDate, toDate)  # get data 
X = X.reset_index()
X = createLags(X, load_lag)                         # create lags

#######################################################################################################################
# Create train/test datasets

split_date = '2023-03-15'                           # date of split 
X_train, X_test = split_in_time(X,split_date)


#######################################################################################################################
# Create model: 
    # For 6 hour model set up max_prediction_length = 6 
    # For 36 hour model set up max_prediction_length = 36 
# For both models the same features was used

max_prediction_length = 6                           # prediction length
max_encoder_length = 25                             # encoder length 
training_cutoff = X_train.index.max() - max_prediction_length

training = TimeSeriesDataSet(
    X_train[lambda x: x.index <= training_cutoff],
    target="Actual Load",
    time_idx= 'time_idx',
    group_ids = ["country"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_categoricals=[],
    time_varying_known_reals=["timestamp","hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_binary","holiday","holiday_lag", "holiday_lead",
                              "load_lag_24", "load_lag_48", "load_lag_168", 
                              #"fct_temp", 
                              'weekday'],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "01_temp", "02_temp", "03_temp","04_temp","05_temp",
        "06_temp","07_temp","08_temp","09_temp","10_temp",
        "08_tsun",
        "01_dwpt", "02_dwpt", "03_dwpt","04_dwpt","05_dwpt",
        "06_dwpt","07_dwpt","08_dwpt","09_dwpt","10_dwpt"

    ], 
    # + Oblacnost/slunecni svetlo 
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, X_train, predict=True, stop_randomization=True)


# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

import warnings
warnings.filterwarnings("ignore")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
##logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=200,
    accelerator="cpu", 
    ##accelerator="mps",
    enable_model_summary=True,
    gradient_clip_val=0.02,
    #limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback]
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate= 0.001,
    hidden_size=30,
    attention_head_size=1,
    dropout=0.3,
    hidden_continuous_size=25,
    lstm_layers = 2,
    loss=QuantileLoss(),
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


#######################################################################################################################
# Optimal Hyperparameters Search
#           For 6 hour model: learning_rate = 0.001, hidden_size = 30, attention_head_size = 1, dropout_rate = 0.3, continuous_hidden_size = 25
#           For 36 hour model: learning_rate = 0.001, hidden_size = 60, attention_head_size = 2, dropout_rate = 0.7, continuous_hidden_size = 30

if search_hyperparameters: 
# create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

#######################################################################################################################
# Model training

if train: 
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
else:
    best_tft = TemporalFusionTransformer.load_from_checkpoint('C:/Users/Yana.Podlesna/skoool/optuna_test/trial_19/epoch=18.ckpt')

#######################################################################################################################
# Results for validation set

country = 'CZ'                                  #country to predict
X = X.loc[X['country'] == country]

raw_prediction = best_tft.predict(
    training,
    mode="raw",
    return_x=True,
)
best_tft.plot_prediction(raw_prediction.x, raw_prediction.output, idx = 0)

# Predictions vs Actuals by variables
predictions = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

# Attention, Variable Selection Network 
interpretation = best_tft.interpret_output(raw_prediction.output, reduction="sum")
best_tft.plot_interpretation(interpretation)


#######################################################################################################################
# Prediction on the test data 

predicted_data = tft_predict(X, X_train, max_prediction_length, max_encoder_length, best_tft)
predicted_data.to_csv('TFT_prediction.csv')

#######################################################################################################################
# Model Evaluation

printMetrics(predicted_data)
printMetricsHourly(predicted_data)


