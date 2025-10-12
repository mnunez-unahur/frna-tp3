import numpy as np
from PIL import Image


# convierte un valor hexa en una matrix de 5 bits x 7elementos
def hexa_a_matriz(valores, ancho_bits=5):
    res = []
    for valor in valores:
        # paso el valor a binario
        b = bin(valor)

        # elimino la parte inicial y relleno con ceros
        b = b[2:].zfill(ancho_bits)

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
    ruido = np.random.normal(0, desviacion, len(entrada))
    salida = entrada.copy()+ruido
   
    # salida = entrada.copy()
    #
    # for j in range(len(entrada)):
    #     r = np.random.random()
    #     if r < 0.02:
    #         salida[j] = 1 if salida[j] == 0 else 0


    return salida


def cargarImagenes():
    matriz = []

    for i in range(1,11):
        num = str(i).zfill(2)
        imagen = Image.open(f'imagenes/imagen-{num}.png').convert('L').resize((16,16))

        m = np.array(imagen).reshape(256)
        m = np.where(m>0, 1,0)
        matriz.append(m)

    return matriz
