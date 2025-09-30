import numpy as np
import matplotlib.pyplot as plt

from datos import font1
from util import hexa_a_matriz, a_bit
from keras import models, layers, optimizers, metrics, utils, activations
X = []
for d in font1:
    # print(f"convirtiendo {d} a matriz")
    m = hexa_a_matriz(d).reshape(35)
    # print(m)
    X.append(m)
X = np.array(X)

utils.set_random_seed(2)


autoencoder1 = models.Sequential()

n = 0.002
epocas = 10000
activacion_latente = 'sigmoid'

# autoencoder1.add(layers.Dense(35,activation = 'sigmoid', input_dim=35))
autoencoder1.add(layers.Input(shape=(35,)))
# autoencoder1.add(layers.Dense(35,activation = 'sigmoid'))
autoencoder1.add(layers.Dense(17,activation = 'sigmoid'))
autoencoder1.add(layers.Dense(2, activation = activacion_latente, name='latente'))
autoencoder1.add(layers.Dense(17,activation = 'sigmoid'))
autoencoder1.add(layers.Dense(35,activation = 'sigmoid'))


opt = optimizers.Adam(learning_rate = n)
metrica = metrics.BinaryAccuracy(name="binary accuracy", dtype=None, threshold=0.5)

# autoencoder1.compile(loss = 'MSE', optimizer = opt, metrics = [metrica])
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

filename = str(n)+"_"+str(epocas)+"_"+activacion_latente

print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")
if cantErrores == 0:
    autoencoder1.save(filename+".keras")


#np.save(filename+".npy", historia.history)
#history = np.load(filename+".npy", allow_pickle="TRUE").item()

plt.plot(historia.history['loss'])
plt.title(f"Entrenamiento n:{n}")
plt.xlabel("pérdida")
plt.ylabel("épocas")
plt.legend(["entrenamiento"],loc="upper right")
plt.savefig(filename+".png")

