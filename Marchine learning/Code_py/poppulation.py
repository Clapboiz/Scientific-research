import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cities = pd.read_csv("california_cities.csv")
cities.head()
print(cities)

#lat: vi do, lon: kinh do
lat, lon = cities["latd"], cities["longd"]
population, area = cities["population_total"], cities["area_total_km2"]

#OO API pylot
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(8,6))
plt.scatter(lon, lat,
            c=np.log10(population), cmap="viridis",
            s=area, linewidths=0, alpha=0.5)
plt.axis("equal")
plt.xlabel("lat")
plt.ylabel("long")
plt.title("Population: cali")

#Create a legend for cities'sizes
area_range = [50, 100, 300, 500]

for area in area_range:
    plt.scatter([], [],  s=area, c='k', alpha=0.4,
                label=str(area) + ' km$^2$')

plt.legend(labelspacing=1, title='City Area')


plt.title('California Cities: Population and Area Distribution');
plt.show()
