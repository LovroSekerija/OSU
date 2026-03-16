import pandas as pd
import matplotlib.pyplot as plt

path = "C:/Users/student/Desktop/lv3/data_C02_emission.csv"
df = pd.read_csv(path)

# --------------------------------------------
# a)
# -------------------------------------------

plt.figure()

plt.hist(df["CO2 Emissions (g/km)"], bins=12, color="lightblue", edgecolor="black")

plt.title("Histogram CO2 emisija")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Broj vozila")

plt.show()

# --------------------------------------------
# b)
# -------------------------------------------

plt.figure()

fuel_types = df["Fuel Type"].unique()

for fuel in fuel_types:
    subset = df[df["Fuel Type"] == fuel]
    
    plt.scatter(
        subset["Fuel Consumption City (L/100km)"],
        subset["CO2 Emissions (g/km)"],
        label=fuel
    )

plt.title("Odnos gradske potrošnje i CO2 emisije")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Fuel Type")

plt.show()

# --------------------------------------------
# c)
# -------------------------------------------

plt.figure()

fuel_types = df["Fuel Type"].unique()

data = []
for fuel in fuel_types:
    data.append(df[df["Fuel Type"] == fuel]["Fuel Consumption Hwy (L/100km)"])

plt.boxplot(data, labels=fuel_types)

plt.title("Izvangradska potrošnja po tipu goriva")
plt.xlabel("Fuel Type")
plt.ylabel("Fuel Consumption Hwy (L/100km)")

plt.show()

# --------------------------------------------
# d)
# --------------------------------------------

fuel_counts = df.groupby("Fuel Type").size()

plt.figure()

plt.bar(fuel_counts.index, fuel_counts.values)

plt.title("Broj vozila po tipu goriva")
plt.xlabel("Fuel Type")
plt.ylabel("Broj vozila")

plt.show()

# --------------------------------------------
# e)
# --------------------------------------------

avg_co2 = df.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()

plt.figure()

plt.bar(avg_co2.index, avg_co2.values)

plt.title("Prosječna CO2 emisija po broju cilindara")
plt.xlabel("Cylinders")
plt.ylabel("Average CO2 Emissions (g/km)")

plt.show()