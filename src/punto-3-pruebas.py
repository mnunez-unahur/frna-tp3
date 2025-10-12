import numpy as np
import matplotlib.pyplot as plt

from datos import font1
from util import hexa_a_matriz, a_bit, distorsionar, cargarImagenes
from keras import models, layers, optimizers, metrics, utils, activations


X = cargarImagenes()
Y = X.copy()
cantMuestras = len(X)

cantidad_diferentes = 0
for x in range(cantMuestras):
    for i in range(1):
        con_ruido = distorsionar(X[x], 0.2)
        X.append(con_ruido)
        Y.append(X[x].copy())

        if not np.array_equal(X[x], con_ruido):
            print(f"entrada         : {X[x]}")
            print(f"entrada con ruido: {con_ruido}")
            cantidad_diferentes += 1

print(f"cantidad de muestras con ruido: {cantidad_diferentes}/{len(X)}")
       

X = np.array(X)
Y = np.array(Y)


codificador=models.load_model("punto-3-codificador.keras")
decodificador=models.load_model("punto-3-decodificador.keras")
autoencoder1 = models.Sequential([codificador,decodificador])

#res = a_bit(autoencoder1.predict(X))
res = autoencoder1.predict(X)

cantErrores = 0
for i in range(len(X)):
    x = X[i].reshape(16,16)
    y_esperada = Y[i].reshape(16,16)
    y_obtenida = res[i].reshape(16,16)

    num = str(i).zfill(3)

    plt.clf()
    plt.imshow(x,interpolation="nearest")
    plt.savefig(f"punto-3-{num}-1-x.png")

    plt.clf()
    plt.imshow(y_esperada,interpolation="nearest")
    plt.savefig(f"punto-3-{num}-2-y_esperada.png")

    plt.clf()
    plt.imshow(y_obtenida,interpolation="nearest")
    plt.savefig(f"punto-3-{num}-3-y_obtenida_{i}.png")


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
plt.savefig(f"punto-2-predict-latente.png")
plt.clf() # limpio la imagen



