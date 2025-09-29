import numpy as np

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
# Y = np.eye(32)

utils.set_random_seed(2)

autoencoder1 = models.Sequential()
# autoencoder1.add(layers.Input(shape=(35,)))

relu = lambda x: activations.relu(x, threshold=0.5, max_value=1)
autoencoder1.add(layers.Dense(35,activation = 'sigmoid', input_dim=35))
autoencoder1.add(layers.Dense(17,activation = 'sigmoid'))
autoencoder1.add(layers.Dense(2, activation = 'sigmoid', name='latente'))
autoencoder1.add(layers.Dense(17,activation = 'sigmoid'))
autoencoder1.add(layers.Dense(35,activation = 'sigmoid'))

opt = optimizers.Adam(learning_rate = 0.008)
metrica = metrics.BinaryAccuracy(name="binary accuracy",
          dtype=None, threshold=0.5)

capa_latente = autoencoder1.layers[3]
print(capa_latente)

# autoencoder1.compile(loss = 'MSE', optimizer = opt, metrics = [metrica])
autoencoder1.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = [metrica])

historia = autoencoder1.fit(X, X, epochs = 3000, batch_size = 1)


res = a_bit(autoencoder1.predict(X))

for i in range(len(X)):
    # x_1 = np.array([X[i, :]])
    # y_1 = a_bit(autoencoder1.predict(x_1))
    if not np.array_equal(X[i], res[i]):
        print("--------------------------------------   ")
        print (X[i])
        print(res[i])





