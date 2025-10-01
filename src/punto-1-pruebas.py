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


codificador=models.load_model("codificador.keras")
decodificador=models.load_model("decodificador.keras")
autoencoder1 = models.Sequential([codificador,decodificador])

res = a_bit(autoencoder1.predict(X))

cantErrores = 0
for i in range(len(X)):
    if not np.array_equal(X[i], res[i]):
        print("--------------------------------------   ")
        print (X[i])
        print(res[i])
        cantErrores += 1


print(f"porcentaje de error: {cantErrores * 100 / len(X)}%")

resultado_latente=codificador.predict(X)

print(resultado_latente)
x_latente = [p[0] for p in resultado_latente]
y_latente = [p[1] for p in resultado_latente]

plt.figure(1)
plt.plot(resultado_latente)
plt.title("predict: latente")
plt.xlabel("x")
plt.ylabel("y")
plt.legend("legend",loc="upper right")
plt.figure(figsize=(8,6))
plt.scatter(x_latente, y_latente)
plt.savefig("punto-1-predict-latente.png")
plt.clf() # limpio la imagen

result = a_bit(decodificador.predict(np.array([[5.47945082e-01, 6.63289547e-01]])))
# result = a_bit(decodificador.predict(np.array([[5, 6]])))
print(result[0].reshape(7,5))

plt.figure(2)
plt.imshow(result[0].reshape(7,5),interpolation="nearest")
plt.savefig("punto-1-predict-resultado.png")

