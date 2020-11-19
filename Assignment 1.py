"""Práctica 1 - Visión por computadora
   Brandon Francisco Hernández Troncoso
   27 - enero - 2019"""

#Imports necesarios
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as io
from skimage.transform import resize

#Pregunta 1: Leer la imagen, guardarla en memoria y escalarla a un tamaño de
#200 x 200 pixeles.
print ("Pregunta 1")
img = io.imread('Lena-grayscale.jpg')
img=resize(img,(200,200))
io.imshow(img)
plt.axis('off')
plt.show()

#Pregunta 2: Crear una máscara de valores booleanos true o false donde 
#la condición sea true para cada pixel que sea mayor o igual al 80% 
#de la mayor intensidad de la imagen.
print("")
print("Pregunta 2")
masc8=img>=img.max()*.8
print (masc8)

#Pregunta 3: Crear una copia de Lena donde los valores de pixeles iguales o
#mayores al 80% del valor máximo mantengan su valor y el resto valgan cero.
print("")
print ("Pregunta 3")
mascLena=img.copy()
mascLena[img<img.max()*.8]=0
io.imshow(mascLena)
plt.axis('off')
plt.show()

#Pregunta 4: Filtro pasa altas

#Pregunta 5: Crear una función que reciba como parámetros una imágen en
#escala de grises y un kernel y que devuelva su convolución.
#Asumiendo que el kernel recibido será una matriz de tamaño mxn, se buscará que
#la convolución se logre con un kernel de tamaño mxm.
def convolve2d(image, kernel):
    m,n=kernel.shape
    if(m!=n):
        maxi=max(m,n)
        if (maxi%2==0):
            maxi=maxi+1
        k2=np.zeros((maxi,maxi))
        k2[:m,:n]=kernel[:m,:n]
    else:
        maxi=max(m,n)
        if(m%2==0):
            maxi=m+1
        k2=np.zeros((maxi,maxi))
        k2[:m,:m]=kernel[:m,:m]
    
    k2=np.flip(k2)
    m,n=k2.shape
    x,y=image.shape
    res=int(m/2)
   
    imageAux=np.zeros((x+m-1,y+m-1))
    
    for j in range (y):
        for i in range (x):
            imageAux[res+i,res+j]=image[i,j]
    
    new_image=np.zeros_like(image)
    
    i=0
    j=0
    l=0
    k=0
    for j in range (y):
        for i in range (x):
            suma=0
            for l in range(m):
                for k in range(m):
                    suma=suma+k2[k,l]*imageAux[i+k,j+l]
                k=0
            l=0
            new_image[i,j]=abs(suma)
            suma=0
        i=0
                    
    
    return new_image


#Pregunta 6: Calculo de la convolución de Lena con el kernel:
#  |-1 -1 -1|
#k=| 2  2  2|
#  |-1 -1 -1|

k=np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])    
convolucionLena=convolve2d(img,k)

#Pregunta 7: Mostrar el resultado
print ("")
print ("Pregunta 6 y 7")
plt.imshow(convolucionLena, cmap='gray')
plt.axis('off')
plt.show()

#Pregunta 8: ¿Qué pasa si reemplazas k por su transpuesta?
k=np.transpose(k)
convolucionLena=convolve2d(img,k)

#Pregunta 9: Muestra el resultado
print ("")
print ("Preguntas 8 y 9")
plt.imshow(convolucionLena, cmap='gray')
plt.axis('off')
plt.show()

#Pregunta 10: muestra el resultado de aplicar 1 y 4 veces consecutivas el
#siguientes kernel gaussiano

gauss=(1/256)*np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
print ("")

"""una vez"""
print("")
print("Pregunta 10: kernel gaussiano 1 vez")
convolucionLena=convolve2d(img,gauss)
plt.imshow(convolucionLena, cmap='gray')
plt.axis('off')
plt.show()

"""tres veces más"""
print("")
print("Pregunta 10: kernel gaussiano 4 veces")
for i in range(0,3):
    convolucionLena=convolve2d(convolucionLena, gauss)
plt.imshow(convolucionLena, cmap='gray')
plt.axis('off')
plt.show()

#pregunta 11: ¿cuántas veces se tiene que hacer el filtro gaussiano para que
#la variación promedio entre los pixeles sea de .1?
print("")
print("Pregunta 11: No converge mi desviación estandar, da overflow después de 3200 iteraciones")
"""count=0
while(np.std(convolucionLena)>.1):
    count=count+1
    convolucionLena=convolve2d(convolucionLena,gauss)
print (count)
"""
#Pregunta 12: aplicar el kernel del punto 6 y luego el kernel gaussiano 
print("")
print("Pregunta 12: ")
nvaConvolucion=convolve2d(img,gauss)
nvaConvolucion=convolve2d(nvaConvolucion,np.transpose(k))
plt.imshow(nvaConvolucion, cmap='gray')
plt.axis('off')
plt.show()

#Pregunta 15:
pregunta15=convolve2d(img,k)
i=0
for i in range(3):
    pregunta15=convolve2d(pregunta15,k)

resultado=np.zeros_like(img)
i=0
j=0

for j in range (200):
    for i in range (200):
        resultado[i,j]=pregunta15[i,j]-img[i,j]

plt.imshow(resultado, cmap='gray')
plt.axis('off')
plt.show()
