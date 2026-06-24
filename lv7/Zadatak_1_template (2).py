import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# Zadatak 1: Mijenjanje načina generiranja podataka
# a)
flagc = [1,2,3,4,5]
for flag in flagc: 
    # generiranje podatkovnih primjera
    X = generate_data(500, flag)

    # prikazi primjere u obliku dijagrama rasprsenja
    plt.figure()
    plt.scatter(X[:,0],X[:,1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'podatkovni primjeri za flag = [{flag}]')
    plt.show()

#b)
k_values = [2, 3, 4, 6, 10]  # 5 različitih vrijednosti K

for flag in flagc:
    X = generate_data(500, flag)
    
    # Kreiramo pod-grafove (subplots) za svaku vrijednost K unutar istog flaga
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Eksperiment s različitim K za FLAG = {flag}', fontsize=16)

    for i, k in enumerate(k_values):
        # Inicijalizacija i treniranje K-means
        km = KMeans(n_clusters=k, init="random", n_init=5, random_state=0)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        # Crtanje rezultata na odgovarajući subplot
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
        axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
        axes[i].set_title(f'K = {k}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#c) 
n_clusters_map = {1: 3, 2: 3, 3: 4, 4: 2, 5: 2}
for flag in flagc: 
    X = generate_data(500, flag)
    
    # K-means algoritam
    # Ovdje koristimo mapirani "optimalni" broj grupa
    k = n_clusters_map[flag]
    km = KMeans(n_clusters=k, init="random", n_init=5, random_state=0)
    labels = km.fit_predict(X)

    # Prikaz rezultata
    plt.figure(figsize=(8, 6))
    # Ključno: c=labels boji točke, cmap određuje paletu boja
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolors='k')
    
    # Prikaz središta grupa (centroida)
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroidi')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'K-means grupiranje (K={k}) za flag = {flag}')
    plt.legend()
    plt.show()

