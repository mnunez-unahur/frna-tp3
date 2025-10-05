import numpy as np

# convierte un valor hexa en una matrix de 5 bits x 7elementos
def hexa_a_matriz(valores):
    res = []
    for valor in valores:
        # paso el valor a binario
        b = bin(valor)

        # elimino la parte inicial y relleno con ceros
        b = b[2:].zfill(5)

        # agregamos la linea a la matriz
        # l = list(b)
        # res.append(np.astype(l,int))
        l = []
        for i in b:
          l.append(int(i))
        res.append(l)

    return np.array(res)

# recibe un array de flotantes y devuelve 0 o 1
def a_bit(valores, umbral=0.5):
    res = []
    for linea in valores:
        linea_bits = []
        for v in linea:
            if v > umbral:
                linea_bits.append(1)
            else:
                linea_bits.append(0)
        res.append(linea_bits)

    return np.array(res)


def distorsionar(entrada, desviacion=0.2):
    # generamos random gausiano
    ruido = np.random.normal(0, desviacion, len(entrada))

    # le sumamos el ruido a la entrada
    salida = entrada.copy()+ruido

    return salida


