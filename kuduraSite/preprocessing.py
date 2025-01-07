import numpy as np
import pandas as pd
import os
import joblib
from django.conf import settings


scaler_path = os.path.join(settings.BASE_DIR, 'models', 'final_scalerT.pkl')
scaler = joblib.load(scaler_path)

numerical_features = [
    'Apparent_Power(kVA)',
    'Power_Factor(PF)',
    'Power(kW)',
    'Reactive_Power(kVAr)',
    'voltage(V)',
    'Current(A)',
    'Energy(kWh)'
]

time_features = [
    'hour_sin',
    'hour_cos',
    'minute_sin',
    'minute_cos'
]

feature_columns = numerical_features + time_features


def preprocess_live_data(new_data, new_timestamp):
    df_new = pd.DataFrame([new_data])

    if df_new.isnull().values.any():
        df_new = df_new.fillna(method='ffill')

    df_new['hour'] = new_timestamp.hour
    df_new['minute'] = new_timestamp.minute
    df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
    df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
    df_new['minute_sin'] = np.sin(2 * np.pi * df_new['minute'] / 60)
    df_new['minute_cos'] = np.cos(2 * np.pi * df_new['minute'] / 60)

    df_new = df_new[feature_columns]

    X_new_scaled = scaler.transform(df_new)

    return X_new_scaled