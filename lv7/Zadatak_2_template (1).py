import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
import os

# ucitaj sliku
img = Image.imread("imgs/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# dobivanje kolicine boja, odnosno elemenata slike: 
print(f"Velicina polja originalne slike: {img_array.shape}")
print(f"Ukupna broj piksela originalne slike: {w * h}")
# rezultatna slika
img_array_aprox = img_array.copy()

# 1. Definiraj broj boja (K)
K = 8  # npr. želimo svesti sliku na samo 8 boja
km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)

# 2. Pokreni grupiranje nad pikselima
# Algoritam će pronaći K reprezentativnih RGB boja (centroida)
km.fit(img_array)
labels = km.predict(img_array)
centers = km.cluster_centers_

# 3. Zamijeni svaki piksel s bojom pripadajućeg centroida
#labels sadrži indeks (0 do K-1) za svaki piksel, a centers su RGB vrijednosti
img_array_aprox = centers[labels]
print(f"Velicina polja aproksimirane slike: {img_array_aprox.shape}")
print(f"Ukupna broj piksela originalne slike: {w * h}")

# 4. Vrati sliku u originalni oblik (W x H x 3)
img_aprox = np.reshape(img_array_aprox, (w, h, d))

# 5. Prikaži aproksimiranu sliku
plt.figure()
plt.title(f"Aproksimirana slika s {K} boja")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# 2. Testiranje različitih vrijednosti K
K_values = [2, 5, 12, 20] # Možeš mijenjati ove brojeve

# Kreiranje prikaza (Subplots)
fig, axes = plt.subplots(1, len(K_values) + 1, figsize=(20, 5))

# Prikaz originala na prvom mjestu
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')

for i, K in enumerate(K_values):
    # K-means algoritam
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    labels = km.fit_predict(img_array)
    centers = km.cluster_centers_
    
    # Zamjena piksela s pripadajućim centrima
    img_array_aprox = centers[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))
    
    # Prikaz aproksimacije
    axes[i+1].imshow(img_aprox)
    axes[i+1].set_title(f"K = {K}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()

images = []

for i in range(1,7):
    images.append(Image.imread(f"imgs/imgs/test_{i}.jpg"))

print(f"Trenutno je zauzeto: {len(images)}  slika")


# --- OBRADA I PRIKAZ ---
K = 8 # Broj boja
n = len(images)

# Kreiramo prozor s n redaka i 2 stupca
fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))

for i in range(n):
    # 1. Uzmi sliku i pretvori u float64 [0, 1]
    img = images[i].astype(np.float64) / 255
    w, h, d = img.shape
    
    # 2. Reshape u 2D niz piksela
    img_array = np.reshape(img, (w * h, d))
    
    # 3. K-means algoritam
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    labels = km.fit_predict(img_array)
    centers = km.cluster_centers_
    
    # 4. Zamjena boja i vraćanje u 3D oblik slike
    img_aprox = centers[labels].reshape(w, h, d)
    
    # 5. Crtanje
    # Original (lijeva strana)
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f"Original {i+1}")
    axes[i, 0].axis('off')
    
    # Rezultat (desna strana)
    axes[i, 1].imshow(img_aprox)
    axes[i, 1].set_title(f"K-means (K={K})")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()

vrijednosti_k = range(1, 11)  # Testiramo K od 1 do 10
inercije = []

for k in vrijednosti_k:
    km = KMeans(n_clusters=k, init="random", n_init=10, random_state=0)
    km.fit(img_array) # Zamijeni s img_array ako radiš sa slikom
    inercije.append(km.inertia_)

# Grafički prikaz
plt.figure(figsize=(8, 5))
plt.plot(vrijednosti_k, inercije, marker='o', linestyle='--')
plt.xlabel('Broj grupa (K)')
plt.ylabel('Inercija (J)')
plt.title('Metoda lakta za određivanje optimalnog K')
plt.xticks(vrijednosti_k)
plt.grid(True, alpha=0.3)
plt.show()

fig, axes = plt.subplots(1,K, figsize=(15, 5))

for i in range(K):
    # 1. Kreiramo binarnu masku za i-tu grupu
    # labels == i vraća True za piksele koji pripadaju grupi i, inače False
    maska = (labels == i)
    
    # 2. Vraćamo masku u 2D oblik slike (W x H)
    binarna_slika = np.reshape(maska, (w, h))
    
    # 3. Prikazujemo binarnu sliku (cmap='gray' jer je binarna)
    axes[i].imshow(binarna_slika, cmap='gray')
    axes[i].set_title(f"Grupa {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()