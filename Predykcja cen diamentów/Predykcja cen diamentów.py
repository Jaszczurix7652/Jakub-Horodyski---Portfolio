import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Wczytanie danych
diamonds = pd.read_csv('diamonds.csv')

# Usunięcie rekordów z zerowymi wartościami 'x', 'y', 'z'
diamonds = diamonds[(diamonds["'x'"] != 0) & (diamonds["'y'"] != 0) & (diamonds["'z'"] != 0)]

# Zmapowanie kolumn kategorycznych
label_encoder = LabelEncoder()
diamonds['cut'] = label_encoder.fit_transform(diamonds['cut'])
diamonds['color'] = label_encoder.fit_transform(diamonds['color'])
diamonds['clarity'] = label_encoder.fit_transform(diamonds['clarity'])

# Tworzenie nowych cech
diamonds['xy_multiply'] = diamonds["'x'"] * diamonds["'y'"]
diamonds['carat_depth_multiply'] = diamonds['carat'] * diamonds['depth']
diamonds['xy_diff'] = diamonds["'x'"] - diamonds["'y'"]
diamonds['xy_sum'] = diamonds["'x'"] + diamonds["'y'"]

# Dzielenie danych na zestawy treningowe i testowe
X = diamonds.drop('price', axis=1)
y = diamonds['price']
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

# Wybór modelu i dopasowanie
#===================================================
#UCZENIE ZWYKLE
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
#UCZENIE Z LOGARYTMEM
model_log = RandomForestRegressor(n_estimators=300, random_state=42)
model_log.fit(X_train, np.log1p(y_train))
#UCZENIE Z PIERWIASTKIEM 4
model_sqrt4 = RandomForestRegressor(n_estimators=300, random_state=42)
model_sqrt4.fit(X_train, np.power(y_train,1/4))
#======================
# Predykcja na danych testowych
#======================
#ZWYKŁA
predicted_prices = model.predict(X_test)
#======================
#Z LOGARYTMEM
predicted_prices_log = model_log.predict(X_test)
# Odwrócenie przewidywanych cen zlogarytmowanych
predicted_prices_log = np.expm1(predicted_prices_log)
#Z PIERWIASTKIEM 4
#======================
predicted_sqrt4 = model_sqrt4.predict(X_test)
# Odwrócenie przewidywanych cen pierwiastkowanych
predicted_sqrt4 = np.power(predicted_sqrt4,4)
#WYKRESY
#===================================================
#ZWYKLE
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted_prices, alpha=0.5)
plt.xlabel("Rzeczywista cena")
plt.ylabel("Przewidywana cena ")
plt.title("Porównanie rzeczywistej ceny z przewidywaną [ZWYKLE]")
plt.show()

plt.subplot(1, 2, 2)
plt.hist([y_test, predicted_prices], bins=20, label=['Rzeczywista cena', 'Przewidywana cena'])
plt.legend()
plt.title('Histogram rzeczywistej i przewidywanej ceny')
plt.tight_layout()
plt.show()

#Z LOGARYTMEM
# Wykres porównujący rzeczywiste ceny z przewidywanymi cenami po odwróceniu zlogarytmowania
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted_prices_log, alpha=0.5)
plt.xlabel("Rzeczywista cena")
plt.ylabel("Przewidywana cena ")
plt.title("Porównanie rzeczywistej ceny z przewidywaną [LOGARYTMOWANIE]")
plt.show()

plt.subplot(1, 2, 2)
plt.hist([y_test, predicted_prices_log], bins=20, label=['Rzeczywista cena', 'Przewidywana cena'])
plt.legend()
plt.title('Histogram rzeczywistej i przewidywanej ceny')
plt.tight_layout()
plt.show()

#Z PIERWIASTKIEM 4
plt.figure(figsize=(8, 6))
plt.scatter(y_test,predicted_sqrt4, alpha=0.5)
plt.xlabel("Rzeczywista cena")
plt.ylabel("Przewidywana cena ")
plt.title("Porównanie rzeczywistej ceny z przewidywaną [PIERWIASTEK 4]")
plt.show()

plt.subplot(1, 2, 2)
plt.hist([y_test, predicted_sqrt4], bins=20, label=['Rzeczywista cena', 'Przewidywana cena'])
plt.legend()
plt.title('Histogram rzeczywistej i przewidywanej ceny')
plt.tight_layout()
plt.show()

mse_zwykly = mean_squared_error(y_test, predicted_prices)
r2_zwykly = r2_score(y_test, predicted_prices)

mse_logarytmowane = mean_squared_error(y_test, predicted_prices_log)
r2_logarytmowane = r2_score(y_test, predicted_prices_log)

mse_sqrt4 = mean_squared_error(y_test, predicted_sqrt4)
r2_sqrt4 = r2_score(y_test, predicted_sqrt4)

print(f"Mean Squared Error (zwykly): {mse_zwykly}")
print(f"R2 Score (zwykly): {r2_zwykly}")
print("------------------------------")
print(f"Mean Squared Error (logarytmowany): {mse_logarytmowane}")
print(f"R2 Score (logarytmowany): {r2_logarytmowane}")
print("------------------------------")
print(f"Mean Squared Error (pierwiastkowany): {mse_sqrt4}")
print(f"R2 Score (logarytmowany): {r2_sqrt4}")

import xgboost as xgb

# Inicjalizacja modelu XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42)

# Uczenie modelu
model_xgb.fit(X_train, y_train)

# Predykcja na danych testowych
predicted_prices_xgb = model_xgb.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test,predicted_prices_xgb, alpha=0.5)
plt.xlabel("Rzeczywista cena")
plt.ylabel("Przewidywana cena ")
plt.title("Porównanie rzeczywistej ceny z przewidywaną [XGB]")
plt.show()

plt.subplot(1, 2, 2)
plt.hist([y_test, predicted_prices_xgb], bins=20, label=['Rzeczywista cena', 'Przewidywana cena'])
plt.legend()
plt.title('Histogram rzeczywistej i przewidywanej ceny')
plt.tight_layout()
plt.show()

# Ocena modelu
mse_xgb = mean_squared_error(y_test, predicted_prices_xgb)
r2_xgb = r2_score(y_test, predicted_prices_xgb)
print(f"Mean Squared Error (XGBoost): {mse_xgb}")
print(f"R2 Score (XGBoost): {r2_xgb}")
