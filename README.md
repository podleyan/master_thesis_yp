
# Prediction of energy demand in the power system with deep learning multi-horizon forecasting methods

Abstract: This thesis addresses deep learning methods with a focus on Temporal Fusion Transformer
(TFT) for multi-horizon load prediction in the power transmission system. The TFT model architecture
proposed for time series processing includes dynamic variable selection network, temporal processing
using encoder decoder layer and attention mechanism. The performance of the TFT model is compared
with traditional machine learning models such as XGBoost and Random Forest, and is evaluated based
on forecasts for the daily market (24 hours ahead) and the intraday market (6 hours ahead). The results
show the potential of advanced deep learning methods in improving power system management in the
context of increasing integration of renewable energy sources.


Code for each model can be run separattely by runing with according scripts.
* tft.py executes temporal fusion transformer
* xgb.py executes XGBoost model for 36 hours prediction (using reccursive method)
* randomForest.py executes Random Forest model for 36 hours prediction (using reccursive method)
* rf_xgb_direct.py executes Random Forest and XGBoost models for 6 hours prediction (using direct method)
* emsemble.py executes ensemble model (Random Forest direct model) trained on 6 hour models predictions
  
For each of this scripts was used functions: 
  * For downloading data - entsoe_data.py (electricity load), calendar_data.py (calendar data and holidays), weather.py (weather     
   historical data), weather_forecast.py (weather forecast).
  * For data preparation such as creating new features, cleaning, etc - data.py, for TFT data - tft_functions.py

For selecting best features execute feature_search.py. 

In folder predictions was saved predictions for each model - in case you want retraining ensemble model using this predictions. 
