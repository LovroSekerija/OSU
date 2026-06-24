import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Učitavanje podataka
data = pd.read_csv('data_C02_emission.csv')

# Odabir numeričkih veličina
features_num = [
'Engine Size (L)',
'Cylinders',
'Fuel Consumption City (L/100km)',
'Fuel Consumption Hwy (L/100km)',
'Fuel Consumption Comb (L/100km)',
'Fuel Consumption Comb (mpg)'
]

# 1. KORAK: Kodiranje kategoričke veličine pomoću OneHotEncoder-a
# Inicijalizacija encodera
ohe = OneHotEncoder()

# fit_transform očekuje 2D strukturu, pa koristimo data[['Fuel Type']]
# .toarray() pretvara rijetku matricu (sparse matrix) natrag u standardno polje [cite: 171]
X_cat_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

# Preuzimamo imena novih stupaca (npr. 'Fuel Type_D', 'Fuel Type_E'...) i stvaramo DataFrame
encoded_columns = ohe.get_feature_names_out(['Fuel Type'])
df_encoded = pd.DataFrame(X_cat_encoded, columns=encoded_columns, index=data.index)

# 2. KORAK: Spajanje numeričkih i kodiranih kategoričkih stupaca
X_combined = pd.concat([data[features_num], df_encoded], axis=1)
y = data['CO2 Emissions (g/km)']

# 3. KORAK: Podjela podataka (zadržavamo pandas strukturu kako bismo lakše pronašli redak s greškom)
X_train, X_test, y_train, y_test = train_test_split(
X_combined, y, test_size=0.2, random_state=42
)

# 4. KORAK: Izgradnja i treniranje modela
model_cat = LinearRegression()
model_cat.fit(X_train, y_train)

# Procjena izlazne veličine
y_pred = model_cat.predict(X_test)

# Evaluacija modela
mae_cat = mean_absolute_error(y_test, y_pred)
r2_cat = r2_score(y_test, y_pred)

print("--- REZULTATI MODELA S KATEGORIČKOM VARIJABLOM ---")
print(f"Srednja apsolutna pogreška (MAE): {mae_cat:.2f} g/km")
print(f"Koeficijent determinacije (R2): {r2_cat:.4f}")

# Komentar:
# Vidimo pad pogreške i porast R2 vrijednosti zahvaljujući uključenju vrste goriva.

# 5. KORAK: Pronalazak maksimalne pogreške
# Računanje apsolutne pogreške za svaki primjer u testnom skupu
errors = abs(y_test - y_pred)

# Pronalazak maksimalne pogreške i njezinog originalnog indeksa
max_error = errors.max()
max_error_idx = errors.idxmax()

# Dohvaćanje podataka o tom konkretnom vozilu iz početnog skupa podataka
vehicle_with_max_error = data.loc[max_error_idx]

print(f"\nMaksimalna pogreška u procjeni emisije CO2: {max_error:.2f} g/km")
print("\nModel vozila kod kojeg je pogreška najveća:")
print("-" * 50)
print(vehicle_with_max_error[['Make', 'Model', 'Vehicle Class', 'Fuel Type']])
print("-" * 50)