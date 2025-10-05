import numpy as np
import matplotlib.pyplot as plt

from datos import font1
from util import hexa_a_matriz, a_bit, distorsionar
from keras import models, layers, optimizers, metrics, utils, activations


X = []
Y = []
cantidad_diferentes = 0
for d in font1:
    m = hexa_a_matriz(d).reshape(35)
    X.append(m)
    Y.append(m)
    # agregamos versiones distorsionadas
    for i in range(10):
        con_ruido = distorsionar(m, 0.3)
        X.append(con_ruido)
        Y.append(m)
        if not np.array_equal(m, con_ruido):
            print(f"entrada         : {m}")
            print(f"salida con ruido: {con_ruido}")
            cantidad_diferentes += 1


X = np.array(X)
Y = np.array(Y)





codificador=models.load_model("punto-2-codificador.keras")
decodificador=models.load_model("punto-2-decodificador.keras")
autoencoder1 = models.Sequential([codificador,decodificador])

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    if not np.array_equal(Y[i], res[i]):
        print("--------------------------------------   ")
        print("Entrada:       ", X[i])
        print("Salida Esperada", Y[i])
        print("Salida Obtenida", res[i])
        cantErrores += 1


print(f"cantidad de muestras con ruido: {cantidad_diferentes}/{len(X)}:: {cantidad_diferentes/len(X)*100}%")
print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")


