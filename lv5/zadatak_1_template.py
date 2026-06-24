import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# a)
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", marker="o", label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", marker="x", label="Test")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Skup za ucenje i testiranje")
plt.legend()
plt.tight_layout()

# b)
model = LogisticRegression()
model.fit(X_train, y_train)

# c)
theta0 = float(model.intercept_[0])
theta1 = float(model.coef_[0, 0])
theta2 = float(model.coef_[0, 1])

print("Parametri modela:")
print(f"theta0 = {theta0:.4f}")
print(f"theta1 = {theta1:.4f}")
print(f"theta2 = {theta2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", marker="o", label="Train")

x1_vals = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 200)
if abs(theta2) > 1e-12:
    x2_vals = -(theta0 + theta1 * x1_vals) / theta2
    plt.plot(x1_vals, x2_vals, "r-", linewidth=2, label="Granica odluke")
elif abs(theta1) > 1e-12:
    x1_boundary = -theta0 / theta1
    plt.axvline(x=x1_boundary, color="r", linewidth=2, label="Granica odluke")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Granica odluke logisticke regresije")
plt.legend()
plt.tight_layout()

# d)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

print("\nMatrica zabune (test):")
print(cm)
print(f"Tocnost: {acc:.4f}")
print(f"Preciznost: {prec:.4f}")
print(f"Odziv: {rec:.4f}")

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Matrica zabune (test)")
plt.xlabel("Predvidena klasa")
plt.ylabel("Stvarna klasa")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.colorbar()
plt.tight_layout()

# e) 
correct_mask = y_pred == y_test

plt.figure(figsize=(8, 6))
plt.scatter(
    X_test[correct_mask, 0],
    X_test[correct_mask, 1],
    c="green",
    marker="o",
    label="Dobro klasificirani",
)
plt.scatter(
    X_test[~correct_mask, 0],
    X_test[~correct_mask, 1],
    c="black",
    marker="x",
    label="Pogresno klasificirani",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Rezultati klasifikacije na testnom skupu")
plt.legend()
plt.tight_layout()

plt.show()
