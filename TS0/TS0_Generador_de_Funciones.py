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

# %% Defino Variables Globales
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
T_simulacion = fs/N

#%% Defino la función funcion_senoidal
def funcion_senoidal(ff, nn, vmax = 1, dc = 0, ph = 0, fs = 2, ruido = 1): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje Y), funcion (eje X)
    """    
    # Grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph ) + dc + ruido
    return tt, xx

#%% Función para generar distintos tipos de ruido
def ruido(tipo, long = N, v_min = -1, v_max = 1, desv_med = 0, desv_est = 0.1):
    if tipo == 'random':
        ruido = np.random.randn(long)
    elif tipo == 'uniforme':
        ruido = np.random.uniform(v_min, v_max, long)
    elif tipo == 'normal':
        ruido = np.random.normal(desv_med, desv_est, N)
    return ruido

#%% Llamo a la función función_senoidal
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

t_ruido, x_ruido = funcion_senoidal(ff = 1000, nn = 1000, ruido = ruido(tipo = 'uniforme', v_min=0.01, v_max=0.01))
plt.figure(6)
plt.clf()
plt.title('Figura 6: Seno con ruido')
plt.grid(True)
plt.plot(x_ruido, label = 'Ruido Random', color = 'cornflowerblue')
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
#plt.xlim(0, N)
#plt.ylim(min(ruido_random)-0.1, max(ruido_random)+0.1)
plt.tight_layout()

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