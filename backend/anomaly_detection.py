# anomaly_detection.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

# Load historical data
data = pd.read_csv('data/cleaned_waterquality.csv')

# Train Isolation Forest model
model = IsolationForest(contamination=0.1)
model.fit(data[['ph', 'bod']])

# Save the model
with open('anomaly_model.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)
