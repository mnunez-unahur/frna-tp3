import numpy as np
import matplotlib.pyplot as plt

from datos import font1, font2, font3
from util import hexa_a_matriz, a_bit
from keras import models, layers, optimizers, metrics, utils, activations

codificador=models.load_model("punto-1-codificador.keras")
decodificador=models.load_model("punto-1-decodificador.keras")
autoencoder1 = models.Sequential([codificador,decodificador])


X = []
font = "font2"
for d in font2:
    m = hexa_a_matriz(d).reshape(35)
    X.append(m)
X = np.array(X)

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    x = X[i].reshape(7,5)
    y_esperada = X[i].reshape(7,5)
    y_obtenida = res[i].reshape(7,5)
    if not np.array_equal(X[i], res[i]):
        cantErrores += 1
        resultado = "Falla"

        plt.clf()
        plt.imshow(x,interpolation="nearest")
        plt.savefig(f"punto-1-{font}-{i}-{resultado}-x.png")
        
        plt.clf()
        plt.imshow(y_esperada,interpolation="nearest")
        plt.savefig(f"punto-1-{font}-{i}-{resultado}-y_esperada.png")
        
        plt.clf()
        plt.imshow(y_obtenida,interpolation="nearest")
        plt.savefig(f"punto-1-{font}-{i}-{resultado}-y_obtenida_{i}.png")


print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")


resultado_latente=codificador.predict(X)

x_latente = [p[0] for p in resultado_latente]
y_latente = [p[1] for p in resultado_latente]

plt.clf() # limpio la imagen
plt.figure(1)
plt.plot(resultado_latente)
plt.title("predict: latente")
plt.xlabel("x")
plt.ylabel("y")
plt.legend("capa latente",loc="upper right")
plt.figure(figsize=(8,6))
plt.scatter(x_latente, y_latente)
plt.savefig(f"punto-1-{font}-predict-latente.png")
plt.clf() # limpio la imagen


