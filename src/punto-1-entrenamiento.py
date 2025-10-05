import numpy as np
import matplotlib.pyplot as plt

from datos import font1
from util import hexa_a_matriz, a_bit
from keras import models, layers, optimizers, metrics, utils, activations

X = []
for d in font1:
    m = hexa_a_matriz(d).reshape(35)
    X.append(m)
X = np.array(X)

utils.set_random_seed(2)


n = 0.008
epocas = 3500

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
historia = autoencoder1.fit(X, X, epochs = epocas, batch_size = 1)

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    if not np.array_equal(X[i], res[i]):
        print("--------------------------------------   ")
        print (X[i])
        print(res[i])
        cantErrores += 1

# filename = str(n)+"_"+str(epocas)+"_"+activacion_latente
archivo_base = f"{epocas}_{n}"

print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")
if cantErrores == 0:
    codificador.save("codificador.keras")
    decodificador.save("decodificador.keras")

plt.plot(historia.history['loss'])
plt.title(f"Entrenamiento n:{n} - épocas: {epocas}")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend(["entrenamiento"],loc="upper right")
plt.savefig(f"fit_{epocas}_{n}.png")

