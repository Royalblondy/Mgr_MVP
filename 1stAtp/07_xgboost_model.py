import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Ładowanie danych
df = pd.read_csv('mvp_features_v2.csv')

train_df = df[df['Year'] <= 2015].copy()
val_df   = df[(df['Year'] > 2015) & (df['Year'] <= 2019)].copy()
test_df  = df[df['Year'] >= 2020].copy()

print(f"Trening (do 2015): {len(train_df)} wierszy")
print(f"Walidacja (2016-2019): {len(val_df)} wierszy")
print(f"Test (2020+): {len(test_df)} wierszy")

# Definiujemy co model ma widzieć (X) i co ma zgadnąć (y)
features = [
    'Agri_Area_Lag1',
    'Cattle_Head_Lag1',
    'Cattle_Change_Lag1',
    'Cattle_Density_Lag1'
]
target = 'Risk_Class'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Trening na latach < 2020 (Rekordów: {len(X_train)})")
print(f"Testy na latach 2020+ (Rekordów: {len(X_test)})")

# 3. TRENOWANIE MODELU XGBOOST
model = xgb.XGBClassifier(
    n_estimators=200,      # Więcej drzew
    max_depth=7,           # Głębsze drzewa, żeby wyłapać detale
    learning_rate=0.05,    # Wolniejsza, ale dokładniejsza nauka
    subsample=0.8,         # Uczymy się na losowych fragmentach danych (zapobiega overfittingowi)
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 4. PRZEWIDYWANIE "PRZYSZŁOŚCI"
y_pred = model.predict(X_test)

# 5. RAPORT WYNIKÓW
print("\n--- RAPORT KLASYFIKACJI (Dla lat 2020+) ---")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"Ogólna dokładność (Accuracy): {acc:.2%}")

# 6. WIZUALIZACJA: Macierz Pomyłek
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Wzrost (0)', 'Stabilnie (1)', 'Spadek (2)'],
            yticklabels=['Wzrost (0)', 'Stabilnie (1)', 'Spadek (2)'])
plt.ylabel('Prawda (Rzeczywistość)')
plt.xlabel('Przewidywanie Modelu')
plt.title('Macierz Pomyłek: Jak model radzi sobie z latami 2020+')
plt.savefig('wykres_8_macierz_pomylek.png')

# 7. WAŻNOŚĆ CECH
importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print("\n--- CO BYŁO NAJWAŻNIEJSZE DLA MODELU? ---")
print(importance.sort_values(by='Importance', ascending=False))