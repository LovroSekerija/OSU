import pandas as pd

path = "C:/Users/student/Desktop/lv3/data_C02_emission.csv"
df = pd.read_csv(path)

# --------------------------------------------
# a)
# --------------------------------------------

# broj mjerenja
print("Broj mjerenja:", len(df))

# tip svake veličine
print("\nTipovi podataka:")
print(df.dtypes)

# nedostajuće vrijednosti
print("\nNedostajuće vrijednosti:")
print(df.isnull().sum())

# duplikati
print("\nBroj dupliciranih redaka:", df.duplicated().sum())

# brisanje duplikata
df = df.drop_duplicates()

# brisanje redaka s missing vrijednostima
df = df.dropna()

# konverzija kategoričkih varijabli
categorical_columns = ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("category")

# --------------------------------------------
# b)
# --------------------------------------------

city = "Fuel Consumption City (L/100km)"

print("\n3 vozila s NAJVEĆOM gradskom potrošnjom:")
print(df.nlargest(3, city)[["Make", "Model", city]])

print("\n3 vozila s NAJMANJOM gradskom potrošnjom:")
print(df.nsmallest(3, city)[["Make", "Model", city]])

# --------------------------------------------
# c)
# --------------------------------------------

filtered = df[(df["Engine Size (L)"] >= 2.5) & (df["Engine Size (L)"] <= 3.5)]

print("Broj vozila:", len(filtered))
print("Prosječna CO2 emisija:", filtered["CO2 Emissions (g/km)"].mean())

# --------------------------------------------
# d)
# --------------------------------------------

audi = df[df["Make"] == "Audi"]

print("Broj Audi vozila:", len(audi))

audi4 = audi[audi["Cylinders"] == 4]

print("Prosječna CO2 emisija (Audi, 4 cilindra):",
      audi4["CO2 Emissions (g/km)"].mean())

# --------------------------------------------
# e)
# --------------------------------------------

print("\nBroj vozila po cilindrima:")
print(df["Cylinders"].value_counts())

print("\nProsječna CO2 emisija po cilindrima:")
print(df.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())

# --------------------------------------------
# f)
# --------------------------------------------

diesel = df[df["Fuel Type"] == "D"]
regular = df[df["Fuel Type"] == "X"]

city = "Fuel Consumption City (L/100km)"

print("\nDizel prosjek:", diesel[city].mean())
print("Dizel medijan:", diesel[city].median())

print("\nRegular benzin prosjek:", regular[city].mean())
print("Regular benzin medijan:", regular[city].median())

# --------------------------------------------
# g)
# --------------------------------------------

diesel4 = df[(df["Fuel Type"] == "D") & (df["Cylinders"] == 4)]

max_car = diesel4.loc[diesel4["Fuel Consumption City (L/100km)"].idxmax()]

print("\nVozilo s najvećom potrošnjom:")
print(max_car[["Make", "Model", "Fuel Consumption City (L/100km)"]])

# --------------------------------------------
# h)
# --------------------------------------------

manual = df[df["Transmission"].str.startswith("M")]

print("Broj vozila s ručnim mjenjačem:", len(manual))

# --------------------------------------------
# i)
# --------------------------------------------

corr = df.corr(numeric_only=True)

print("\nKorelacijska matrica:")
print(corr)