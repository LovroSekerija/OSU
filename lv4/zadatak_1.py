from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data_C02_emission.csv')

# postavljanje columns koje cemo ucitavati
features_columns = [
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",                
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)",
    ]

# X predstavljaju podatke koji ce se korisiti za ovisnost s C02
# Y predstavlja podatke koji ce se koristiti kao pocetka izlazna vrijednost za koju zelimo pronaci metodu predikcije 

X = df[features_columns]
Y = df["CO2 Emissions (g/km)"]

# splitanje podataka na train i test 
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=1 )

# crtanje scatter-a za svaki od vrijednosti za treniranje 
for feauter in features_columns:
    plt.scatter(X_train[feauter], y_train, color='blue', label='Skup za testiranje', alpha=0.5, s=60)
    plt.scatter(X_test[feauter], y_test, color='red', label='Skup za testiranje', alpha=0.7, s=60)
    plt.title(f"Ovisnost C02 o: {feauter}")
    plt.xlabel("CO2 Emissions (g/km)")
    plt.ylabel(feauter)
    plt.show()

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.fit_transform(X_test)


# crtanje histograma za svaku od navedenih skaliranih vrijednosti
for feauter in features_columns:
    idx = X.columns.get_loc(feauter)
    plt.hist(X_train_n[:, idx], color="teal")
    plt.hist(X_train[feauter],color='skyblue')
    plt.legend(loc='upper left', fontsize='medium', shadow=True)
    plt.show()

# linear model learning 
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

# predikcija izlazne velicine na skupu podataka za testiranje 
y_test_p = linearModel.predict(X_test_n)
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_test_p, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Idealna procjena')
plt.xlabel('Stvarna emisija CO2 (g/km)')
plt.ylabel('Procijenjena emisija CO2 (g/km)')
plt.title('Stvarne vs. Procijenjene vrijednosti emisije CO2')
plt.legend()
plt.show()

# evaluacija modela na skupu podataka za testiranje pomocu MAE
mse = mean_absolute_error(y_test, y_test_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p)
mape = mean_absolute_percentage_error(y_test, y_test_p)
r2 = r2_score(y_test, y_test_p)

print("\n--- VREDNOVANJE MODELA NA TESTNOM SKUPU ---")
print(f"Srednja kvadratna pogreška (MSE): {mse:.2f}")
print(f"Korijen iz srednje kvadratne pogreške (RMSE): {rmse:.2f}")
print(f"Srednja apsolutna pogreška (MAE): {mae:.2f}")
print(f"Srednja apsolutna postotna pogreška (MAPE): {mape:.4f}")
print(f"Koeficijent determinacije (R2): {r2:.4f}")
