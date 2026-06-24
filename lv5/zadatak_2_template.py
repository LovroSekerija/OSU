import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}


def load_penguins_dataframe():
    local_path = Path(__file__).with_name("penguins.csv")
    if local_path.exists():
        return pd.read_csv(local_path)

    try:
        from palmerpenguins import load_penguins

        return load_penguins()
    except Exception:
        try:
            import seaborn as sns

            return sns.load_dataset("penguins")
        except Exception as exc:
            raise FileNotFoundError(
                "Nije pronaden 'penguins.csv', a nisu dostupni ni paketi 'palmerpenguins' ili 'seaborn'."
            ) from exc

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = load_penguins_dataframe()

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'] = df['species'].map({'Adelie': 0,
                                   'Chinstrap': 1,
                                   'Gentoo': 2}).astype(int)

print(df.info())

# izlazna velicina: species
output_variable = 'species'

# ulazne velicine za osnovni model (2D)
base_input_variables = ['bill_length_mm',
                        'flipper_length_mm']

# prosireni skup ulaznih velicina
extended_input_variables = ['bill_length_mm',
                            'flipper_length_mm',
                            'bill_depth_mm',
                            'body_mass_g']

X_base = df[base_input_variables].to_numpy()
X_extended = df[extended_input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test (isti indeksi za obje varijante ulaznih velicina)
X_base_train, X_base_test, X_ext_train, X_ext_test, y_train, y_test = train_test_split(
    X_base,
    X_extended,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y,
)

# a) Broj primjera po klasi (train/test) stupcasti dijagram
train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)

x = np.arange(len(labels))
width = 0.35
train_heights = [train_counts[np.where(train_classes == i)[0][0]] if i in train_classes else 0 for i in range(len(labels))]
test_heights = [test_counts[np.where(test_classes == i)[0][0]] if i in test_classes else 0 for i in range(len(labels))]

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, train_heights, width=width, label='Train')
plt.bar(x + width / 2, test_heights, width=width, label='Test')
plt.xticks(x, [labels[i] for i in range(len(labels))])
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasi (train/test)')
plt.legend()
plt.tight_layout()

# b) Model logisticke regresije (2 ulazne velicine)
model_base = LogisticRegression()
model_base.fit(X_base_train, y_train)

# c) Parametri modela i razlika prema binarnom problemu
print("\nParametri multiclass modela (2 ulazne velicine):")
print("intercept_:")
print(model_base.intercept_)
print("coef_:")
print(model_base.coef_)

# d) Granice odluke za train skup
plot_decision_regions(X_base_train, y_train, model_base)
plt.xlabel(base_input_variables[0])
plt.ylabel(base_input_variables[1])
plt.title('Granice odluke (logisticka regresija, 2 ulazne velicine)')
plt.legend()
plt.tight_layout()

# e) Klasifikacija testnog skupa i metrike
y_pred_base = model_base.predict(X_base_test)
cm_base = confusion_matrix(y_test, y_pred_base)
acc_base = accuracy_score(y_test, y_pred_base)

print("\nMatrica zabune (test, 2 ulazne velicine):")
print(cm_base)
print(f"Tocnost: {acc_base:.4f}")
print("\nClassification report (test, 2 ulazne velicine):")
print(classification_report(y_test, y_pred_base, target_names=[labels[0], labels[1], labels[2]]))

plt.figure(figsize=(5, 4))
plt.imshow(cm_base, cmap='Blues')
plt.title('Matrica zabune (2 ulazne velicine)')
plt.xlabel('Predvidena klasa')
plt.ylabel('Stvarna klasa')
for i in range(cm_base.shape[0]):
    for j in range(cm_base.shape[1]):
        plt.text(j, i, cm_base[i, j], ha='center', va='center', color='black')
plt.colorbar()
plt.tight_layout()

# f) Dodavanje novih ulaznih velicina i usporedba
model_ext = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
model_ext.fit(X_ext_train, y_train)
y_pred_ext = model_ext.predict(X_ext_test)

cm_ext = confusion_matrix(y_test, y_pred_ext)
acc_ext = accuracy_score(y_test, y_pred_ext)

print("\nRezultati nakon dodavanja ulaznih velicina:")
print(f"Tocnost (2 ulaza): {acc_base:.4f}")
print(f"Tocnost (4 ulaza): {acc_ext:.4f}")

print("\nMatrica zabune (test, 4 ulazne velicine):")
print(cm_ext)
print("\nClassification report (test, 4 ulazne velicine):")
print(classification_report(y_test, y_pred_ext, target_names=[labels[0], labels[1], labels[2]]))

plt.figure(figsize=(5, 4))
plt.imshow(cm_ext, cmap='Greens')
plt.title('Matrica zabune (4 ulazne velicine)')
plt.xlabel('Predvidena klasa')
plt.ylabel('Stvarna klasa')
for i in range(cm_ext.shape[0]):
    for j in range(cm_ext.shape[1]):
        plt.text(j, i, cm_ext[i, j], ha='center', va='center', color='black')
plt.colorbar()
plt.tight_layout()

plt.show()
