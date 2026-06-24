import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("-" * 30)
print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
print("-" * 30)

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


# Za KNN model potrebno je koristiti standardizirane podatke 
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# tocnost 
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"Tocnost KNN(skup za ucenje K:5): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:5): {acc_test_knn:.3f}")
print("-" * 30)
# crtanje granice
plot_decision_regions(X_train_n, y_train, classifier = KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Tocnost: " + "{:0.3f}".format((acc_train_knn))+ " K:5")
plt.tight_layout()
plt.show()


# K = 1 
KNN_model = KNeighborsClassifier(n_neighbors=1)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# tocnost 
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"Tocnost KNN(skup za ucenje K:1): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:1): {acc_test_knn:.3f}")
print("-" * 30)
# crtanje granice
plot_decision_regions(X_train_n, y_train, classifier = KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Tocnost: " + "{:0.3f}".format((acc_train_knn)) + " K:1")
plt.tight_layout()
plt.show()


# K = 100
KNN_model = KNeighborsClassifier(n_neighbors=100)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# tocnost 
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"Tocnost KNN(skup za ucenje K:100): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:100): {acc_test_knn:.3f}")
print("-" * 30)
# crtanje granice
plot_decision_regions(X_train_n, y_train, classifier = KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Tocnost: " + "{:0.3f}".format((acc_train_knn)) + " K:100")
plt.tight_layout()
plt.show()
# Mozemo primijetiti kako je granica sama vezana usko uz tocke, stvara se krivulja,
# ali i kao otok jednog skupa podataka. 

# Zakljucak: 
# Za K=1 jako mala vrijednost, svaka tocka se gleda kao bitna pa ce zbog toga nastati
# otoci, te je ovo overfitting 
# Za K=5 do K=15 ukljucujemo razum, odnosno ovdje se gasi šum odvojenih tocaka i smanjuju se otoci
# Za K=100 uzima se samo velika kolicina podataka i time se zapravo dobiva underfitting

# priprema parametara za GridSearch unakrsnu validaciju
knn = KNeighborsClassifier()
param_grids = {
    "n_neighbors" : range(1,31)
}

# unakrsna validacija sa zadanim parametrima 
grid_search = GridSearchCV(estimator = knn, param_grid= param_grids, cv = 5, scoring= "accuracy")
grid_search.fit(X_train_n,y_train)

print(f"Najbolji hiperparametar: {grid_search.best_params_}")
print(f"Najveca tocnost unakrsne validacije : {grid_search.best_score_:.3f}")
print("-" * 30)
# Automatsko koristenje najboljeg modela
najbolji_knn = grid_search.best_estimator_
acc_test_najbolji = najbolji_knn.score(X_test_n, y_test)
print(f"Tocnost najboljeg modela na testnom skupu: {acc_test_najbolji:.3f}")
# K = 7
KNN_model = KNeighborsClassifier(n_neighbors=7)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# tocnost 
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"Tocnost KNN(skup za ucenje K:7): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:7): {acc_test_knn:.3f}")
print("-" * 30)

# Zakljucak: 
# Hiperparametar je najbolja vrijednost parametra K koju programer sam mora izracunati
# Ona je nuzna za dobivanje najoptimalnijeg modela

odabrani_C = 1.0
odabrani_gamma = 5.0

svm_rbf_model = SVC(kernel='rbf', C=odabrani_C, gamma=odabrani_gamma)

# Treniranje modela na standardiziranim podacima
svm_rbf_model.fit(X_train_n, y_train)

# Predviđanje 
y_pred_train_svm = svm_rbf_model.predict(X_train_n)
y_pred_test_svm = svm_rbf_model.predict(X_test_n)

acc_train_svm = accuracy_score(y_train, y_pred_train_svm)
acc_test_svm = accuracy_score(y_test, y_pred_test_svm)

print(f"Tocnost SVM RBF (skup za ucenje): {acc_train_svm:.3f}")
print(f"Tocnost SVM RBF (skup za testiranje): {acc_test_svm:.3f}")
print("-" * 30)

# Crtanje granice odluke
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_model)
plt.xlabel("Standardizirana Dob")
plt.ylabel("Standardizirana Plaća")
plt.legend(loc="upper left")
plt.title(f"SVM RBF | C={odabrani_C}, gamma={odabrani_gamma} | Točnost: {acc_train_svm:.3f}")
plt.tight_layout()
plt.show()

#Zakljucak: 
# Gamma zapravo kontrolira domet jedne tocke. Sto je gamma vece vrijednosti, to znaci da ce
# doci do overfittinga vrlo lako. 

# C regulacija - kontrola strogosti prema pogreskama
# ukoliko je manja vrijednost tada je model "opusten", odnosno sama granica je općenitija
# Kada je C vece vrijednosti 100 onda je model perfekcionisticki i time dobivamo maksimalnost 
# obuhvacenja tocaka

# Mijenjajuci kernele dobivamo (poly, rbf, linear) zapravo dobivamo uvid koji
# nam najbolje odgovara jer time dobivamo najbolju 2D percepciju uhvacenosti podataka
# Odnosno, kernel je kao leca kroz koji model gleda i na temelju toga predict-a
svm_osnovni = SVC(kernel='rbf')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}

# Postavljanje GridSearchCV objekta (5-fold unakrsna validacija)
grid_search_svm = GridSearchCV(estimator=svm_osnovni, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='accuracy')

# Pokretanje pretrage na standardiziranim podacima
print("Započinjem pretragu za optimalnim parametrima...")
grid_search_svm.fit(X_train_n, y_train)

# 5. Ispis najboljih rezultata
print("-" * 30)
print(f"Optimalni parametri: {grid_search_svm.best_params_}")
print(f"Najveca tocnost (Cross-Validation): {grid_search_svm.best_score_:.3f}")
print("-" * 30)

# Izvlačenje apsolutno najboljeg modela iz pretrage
najbolji_svm_model = grid_search_svm.best_estimator_

# Konačna provjera na odvojenom testnom skupu
y_pred_test_najbolji = najbolji_svm_model.predict(X_test_n)
acc_test_najbolji = accuracy_score(y_test, y_pred_test_najbolji)

print(f"Tocnost najboljeg SVM modela na testnom skupu: {acc_test_najbolji:.3f}")

# Crtanje granice odluke za pobjednički model
plot_decision_regions(X_train_n, y_train, classifier=najbolji_svm_model)
plt.xlabel("Standardizirana Dob")
plt.ylabel("Standardizirana Plaća")
plt.legend(loc="upper left")
plt.title(f"Optimalni SVM | C={grid_search_svm.best_params_['C']}, gamma={grid_search_svm.best_params_['gamma']}")
plt.tight_layout()
plt.show()