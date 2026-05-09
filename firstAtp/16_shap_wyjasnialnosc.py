import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("1. Szybkie odtworzenie modelu hybrydowego (bez wycieku danych)...")
df = pd.read_csv('mvp_features_v2.csv')

czarna_lista = ['World', 'Africa', 'Eastern Africa', 'Middle Africa', 'Northern Africa',
    'Southern Africa', 'Western Africa', 'Americas', 'Northern America',
    'Central America', 'Caribbean', 'South America', 'Asia', 'Central Asia',
    'Eastern Asia', 'Southern Asia', 'South-eastern Asia', 'Western Asia',
    'Europe', 'Eastern Europe', 'Northern Europe', 'Southern Europe',
    'Western Europe', 'Oceania', 'Australia and New Zealand', 'Melanesia',
    'Micronesia', 'Polynesia', 'European Union (27)', 'European Union (28)',
    'Least Developed Countries', 'Land Locked Developing Countries (LLDCs)',
    'Small Island Developing States', 'Low Income Food Deficit Countries',
    'Net Food Importing Developing Countries', 'Annex I countries', 'Non-Annex I countries', 'OECD',
    'Bahrain', 'Malta', 'Iceland']

df_clean = df[~df['Area'].isin(czarna_lista)].copy()
df_clean['Forest_Area_Lag1'] = df_clean.groupby('Area')['Forest_Area'].shift(1)
df_clean = df_clean.dropna()

df_encoded = pd.get_dummies(df_clean, columns=['Area'], dtype=float)
df_encoded['Area_Name'] = df_clean['Area']

features_lstm = [col for col in df_encoded.columns if col not in ['Year', 'Forest_Change_%', 'Area_Name', 'Risk_Class']]
target = 'Forest_Change_%'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_encoded[features_lstm] = scaler_X.fit_transform(df_encoded[features_lstm])
df_encoded[[target]] = scaler_y.fit_transform(df_encoded[[target]])

def create_sequences_with_meta(data, window_size=3):
    X, y, years, areas = [], [], [], []
    for area in data['Area_Name'].unique():
        area_df = data[data['Area_Name'] == area].sort_values('Year')
        for i in range(window_size, len(area_df)):
            X.append(area_df[features_lstm].values[i-window_size:i])
            y.append(area_df[target].values[i])
            years.append(area_df['Year'].values[i])
            areas.append(area)
    return np.array(X), np.array(y), np.array(years), np.array(areas)

X_seq, y_seq, years_seq, areas_seq = create_sequences_with_meta(df_encoded, window_size=3)

mask_lstm_train = years_seq <= 2015
X_train_lstm, y_train_lstm = X_seq[mask_lstm_train], y_seq[mask_lstm_train]

model_lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2), Dense(32, activation='relu'), Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=40, batch_size=32, verbose=0, callbacks=[early_stop])

lstm_preds_df = pd.DataFrame({
    'Area': areas_seq, 'Year': years_seq,
    'LSTM_Prediction': scaler_y.inverse_transform(model_lstm.predict(X_seq, verbose=0)).flatten()
})

df_hybrid = pd.merge(df_clean, lstm_preds_df, on=['Area', 'Year'])
df_hybrid['Area'] = df_hybrid['Area'].astype('category')

train_hybrid = df_hybrid[(df_hybrid['Year'] > 2015) & (df_hybrid['Year'] <= 2019)]
test_hybrid  = df_hybrid[df_hybrid['Year'] >= 2020]

features_xgb = ['Area', 'Forest_Area_Lag1', 'Agri_Area_Lag1', 'Cattle_Head_Lag1', 'Cattle_Change_Lag1', 'Cattle_Density_Lag1', 'LSTM_Prediction']
X_train_xgb = train_hybrid[features_xgb]
y_train_xgb = train_hybrid[target]
X_test_xgb = test_hybrid[features_xgb]

model_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, enable_categorical=True)
model_xgb.fit(X_train_xgb, y_train_xgb)

print("2. Generowanie wykresów SHAP (to może zająć kilkanaście sekund)...")
# Tworzymy explainera dla XGBoost
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_xgb)

# WYKRES 1: SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_xgb, show=False)
plt.title("Jak cechy wpływają na wynik modelu (SHAP Summary)")
plt.tight_layout()
plt.savefig('shap_1_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# WYKRES 2: SHAP Bar Plot (Średnia ważność absolutna)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_xgb, plot_type="bar", show=False)
plt.title("Ogólna ważność cech w decyzjach (SHAP Bar)")
plt.tight_layout()
plt.savefig('shap_2_bar.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nSukces!")
