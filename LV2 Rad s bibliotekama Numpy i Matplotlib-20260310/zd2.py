import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("data.csv",delimiter=",",skiprows=1)
spol = data[:,0].copy()
visina = data[:,1].copy()
masa = data[:,2].copy()
print("broj osoba",data.shape[0])
plt.scatter(visina,masa)
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Odnos visine i mase")
plt.show()
novaVisina = visina[::50]
novaMasa = masa[::50]
plt.scatter(novaVisina,novaMasa)
plt.show()
print(np.min(visina))
print(np.max(visina))
print(np.mean(visina))
muskarci = data[spol == 1]
visinaMuskarci = muskarci[:,1].copy()
print(np.min(visinaMuskarci))
print(np.max(visinaMuskarci))
print(np.mean(visinaMuskarci))
zene = data[spol == 0]
visinaZene = zene[:,1].copy()
print(np.min(visinaZene))
print(np.max(visinaZene))
print(np.mean(visinaZene))
