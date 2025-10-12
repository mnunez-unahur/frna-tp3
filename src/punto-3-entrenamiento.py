import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from datos import emojis,emoticons_20x20 
from util import hexa_a_matriz, a_bit, distorsionar, cargarImagenes
from keras import models, layers, optimizers, metrics, utils, activations





X = cargarImagenes()
Y = X.copy()
cantMuestras = len(X)

cantidad_diferentes = 0
for x in range(cantMuestras):
    for i in range(10):
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



utils.set_random_seed(2)


n = 0.001
epocas = 2000

codificador = models.Sequential()
codificador.add(layers.Input(shape=(256,)))
codificador.add(layers.Dense(128,activation = 'sigmoid'))
# codificador.add(layers.Dense(64,activation = 'sigmoid'))
# codificador.add(layers.Dense(32,activation = 'sigmoid'))
codificador.add(layers.Dense(2, activation = 'sigmoid', name='latente'))

decodificador = models.Sequential()
decodificador.add(layers.Input(shape=(2,)))
# decodificador.add(layers.Dense(32,activation = 'sigmoid'))
# decodificador.add(layers.Dense(64,activation = 'sigmoid'))
decodificador.add(layers.Dense(128,activation = 'sigmoid'))
decodificador.add(layers.Dense(256,activation = 'sigmoid'))

opt = optimizers.Adam(learning_rate = n)
metrica = metrics.BinaryAccuracy(name="binary accuracy", dtype=None, threshold=0.5)


autoencoder1 = models.Sequential([codificador,decodificador])
autoencoder1.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = [metrica])
historia = autoencoder1.fit(X, Y, epochs = epocas, batch_size = 1)

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    x = X[i].reshape(16,16)
    y_esperada = Y[i].reshape(16,16)
    y_obtenida = res[i].reshape(16,16)
    if not np.array_equal(Y[i], res[i]):
        cantErrores += 1
        resultado = "Falla"
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

# filename = str(n)+"_"+str(epocas)+"_"+activacion_latente
archivo_base = f"{epocas}_{n}"

print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")
if cantErrores == 0:
    codificador.save("punto-3-codificador.keras")
    decodificador.save("punto-3-decodificador.keras")


plt.clf()
plt.plot(historia.history['loss'])
plt.title(f"Entrenamiento n:{n} - épocas: {epocas}")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend(["entrenamiento"],loc="upper right")
plt.savefig(f"fit_{epocas}_{n}.png")

