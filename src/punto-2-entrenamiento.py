import numpy as np
import matplotlib.pyplot as plt

from datos import font1
from util import hexa_a_matriz, a_bit, distorsionar
from keras import models, layers, optimizers, metrics, utils, activations

utils.set_random_seed(2)
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

print(f"cantidad de muestras con ruido: {cantidad_diferentes}/{len(X)}")


X = np.array(X)
Y = np.array(Y)



n = 0.008
epocas = 2000

codificador = models.Sequential()
codificador.add(layers.Input(shape=(35,)))
codificador.add(layers.Dense(17,activation = 'sigmoid'))
codificador.add(layers.Dense(2, activation = 'sigmoid', name='latente'))

decodificador = models.Sequential()
decodificador.add(layers.Input(shape=(2,)))
decodificador.add(layers.Dense(17,activation = 'sigmoid'))
decodificador.add(layers.Dense(35,activation = 'sigmoid'))

opt = optimizers.Adam(learning_rate = n)
metrica = metrics.BinaryAccuracy(name="binary accuracy", dtype=None, threshold=0.5)


autoencoder1 = models.Sequential([codificador,decodificador])
autoencoder1.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = [metrica])


# entrenamiento
historia = autoencoder1.fit(X, Y, epochs = epocas, batch_size = 1)

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    if not np.array_equal(Y[i], res[i]):
        print("--------------------------------------   ")
        print("Entrada: ", X[i])
        print("Salida Esperada: ", Y[i])
        print("Salida Obtenida", res[i])
        cantErrores += 1


print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")
if cantErrores == 0:
    codificador.save("punto-2-codificador.keras")
    decodificador.save("punto-2-decodificador.keras")

plt.plot(historia.history['loss'])
plt.title(f"Entrenamiento n:{n} - épocas: {epocas}")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend(["entrenamiento"],loc="upper right")
plt.savefig(f"fit_{epocas}_{n}.png")

