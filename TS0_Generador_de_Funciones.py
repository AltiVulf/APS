"""
Created on Spyder 6

@author: Tomás Altimare Bercovich

Descripción:
------------

Generador de señales: Senoidales y Función
    En este programa vamos a parametrizar y llamar a una función "funcion_senoidal",
    la cual genera una señal senoidal parametrizada a partir de: 
    tt, xx = funcion_senoidal(amplitud, offset, frec, fase, #_muestras, frec_muestreo)

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np 
"""
    La uso para realizar operaciones matemáticas como el seno y para hacer la linea de tiempo tt. 
    Tiene tambien funciones de algebra matricial y manejo de datos tipo array
"""
import matplotlib.pyplot as plt # La uso para imprimir el gráfico del seno

#%% Defino la función funcion_senoidal
def funcion_senoidal(vmax, dc, ff, ph, nn, fs): 
  # funcion_senoidal(amplitud, offset, frec, fase, #_muestras, frec_muestreo)
    
    ts = 1/fs # tiempo/periodo de muestreo
    df = fs/nn # resolución espectral
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn - 1) * ts, nn).flatten() # tiempo equiespaciado
   
    xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    return tt, xx
    
#%% Llamo a la función función_senoidal

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
T_simulacion = fs/N

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 1, ph=0, nn = N, fs = fs)

plt.figure(1)
plt.grid(True)
plt.plot(tt, xx, 'o--')
plt.xlim(0, T_simulacion)

#%% Bonus 1: Ploteo la funcion senoidal con diferentes valores para la ff

t1, x1 = funcion_senoidal (vmax = 2, dc = 0, ff = 500, ph=0, nn = N, fs = fs)
t2, x2 = funcion_senoidal (vmax = 2, dc = 0, ff = 999, ph=0, nn = N, fs = fs)
t3, x3 = funcion_senoidal (vmax = 2, dc = 0, ff = 1999, ph=0, nn = N, fs = fs)
t4, x4 = funcion_senoidal (vmax = 2, dc = 0, ff = 2001, ph=0, nn = N, fs = fs)

plt.figure(2)
plt.grid(True)
plt.plot(t1, x1, 'o--')
plt.xlim(0, T_simulacion)

plt.figure(3)
plt.grid(True)
plt.plot(t2, x2, 'o--')
plt.xlim(0, T_simulacion)

plt.figure(4)
plt.grid(True)
plt.plot(t3, x3, 'o--')
plt.xlim(0, T_simulacion)

plt.figure(5)
plt.grid(True)
plt.plot(t4, x4, 'o--')
plt.xlim(0, T_simulacion)

energia = np.sum(xx ** 2)/N # nota: el operador ** es un exponente

# Para tener una senoidal con energia unitaria, necesito que su amplitud sea raiz de 2

print(energia)

#%% Bonus 2: Señal cuadrada

# def funcion_cuadrada(vmax, vmin, duty): 
#   # funcion_cuadrada()
#     ts = 1/fs # tiempo de muestreo
#     df = fs/nn # resolución espectral
    
#     # grilla de sampleo temporal
#     tt = np.linspace(0, (N-1)*ts, N).flatten()
   
#     xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    
#     return tt, xx

# %% AÑADIR: Conclusiones, analisis y discusion de los graficos... (entregar en jupyter notebook)