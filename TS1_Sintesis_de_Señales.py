"""
Created on Wed Aug 20 2025

@author: Tomás Altimare Bercovich

Descripción:
------------
    En este trabajo práctico se realizó un generador de señales 
senoidales y cuadradas a partir de su parametrización. A partir 
del mismo, se generaron señales con diferentes parámetros con 
el fin de graficarlas, analizarlas y compararlas. A su vez, 
se verificó la ortogonalidad y se graficaron las autocorrelaciones 
entre las mismas. Por último, se demostró la vericidad de una 
igualdad trigonométrica planteada. 

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as waves
from tabulate import tabulate

#%% Defino variables del programa
fs = 1000000 # frecuencia de f_wav (Hz)
ts = 1/fs  # Tiempo entre muestras
N = 1000 # cantidad de muestras
T_simulacion = N/fs
amp = 1 # Normalizo la amplitud de las señales en 1

#%% Defino la función para la señal senoidal
def funcion_senoidal(ff, nn, vmax = 1, dc = 0, ph = 0, fs = 2): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje Y), funcion (eje X)
    """    
    # Grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    return tt, xx

#%% Defino la función para la señal cuadrada
def funcion_cuadrada(ff, nn, duty, vmax = 1, dc = 0,ph = 0, fs = 2):
    """
    funcion_cuadrada(frecuencia, #_muestras, duty, amplitud, offset, fase, frec_muestreo)
    - El duty puede ir de 0 a 100
    - Salida: tt (s), xx (amplitud)
    """
    # Grilla de sampleo temporal 
    n = np.arange(nn)
    tt = n / fs
    duty_frac = duty / 100

    # Fase normalizada en [0,1) (construyo la periodicidad de la funcion)
    fase = ( (ff * n / fs) + (ph / (2*np.pi)) ) % 1
    
    # Grafica arriba cuando fase < duty_frac, abajo en el otro caso
    xx = np.where(fase < duty_frac, vmax, -vmax) + dc
    return tt, xx

#%%############
## Punto (1) ##
###############
#%% Armo las funciones que voy a graficar
# Señal 1
t1, x1 = funcion_senoidal (vmax = amp, dc = 0, ff = 2000, ph = 0, nn = N, fs = fs)
# Señal 2
t2, x2 = funcion_senoidal (vmax = 2*amp, dc = 0, ff = 2000, ph = (np.pi/2), nn = N, fs = fs)
# Señal 3
t3, x3 = funcion_senoidal (vmax = amp, dc = 0, ff = 1000, ph = 0, nn = N, fs = fs)
x_modulada = x1 * x3
# Señal 4
recorte = 0.75
x_recortada = np.clip(x1, -amp*recorte, amp*recorte)
potencia_total = (amp**2)/2
potencia_al_75 = (0.75) * potencia_total
# Señal 5
tsq1, xsq1 = funcion_cuadrada(vmax = amp, dc = 0, ff = 4000, duty = 50, ph = 0, nn = N, fs = fs)
# Señal 6
t_rectangulo = 0.001 # duracion de un periodo
ff_rectangulo = 1/t_rectangulo # periodo
tsq2, xsq2 = funcion_cuadrada(vmax = amp, dc = 0, ff = ff_rectangulo, duty = 70, ph = 0, nn = N, fs = fs)

#%% Imprimo los gráficos
def plotear_fig1_a_fig6 ():
    plt.figure(1)
    plt.title('Figura 1: Seno con frec = 2KHz')
    plt.grid(True)
    plt.plot(t1, x1, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    # Añado una leyenda con los datos sobre la señal:
    potencia = np.sum(x1 ** 2)/(2*N+1)
    info_fig1 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig1, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(2)
    plt.title('Figura 2: Seno con frec = 2KHz desfasada π/2')
    plt.grid(True)
    plt.plot(t2, x2, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    potencia = np.sum(x2 ** 2)/(2*N+1)
    info_fig2 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig2, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(3)
    plt.title('Figura 3: Seno de 2KHz modulada con Seno de 1KHz')
    plt.grid(True)
    plt.plot(t1, x_modulada, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    potencia = np.sum(x_modulada ** 2)/(2*N+1)
    info_fig3 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig3, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(4)
    plt.title('Figura 4: Seno de 2KHz recortado al 75%')
    plt.grid(True)
    plt.plot(t1, x_recortada, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    potencia = np.sum(x_recortada ** 2)/(2*N+1)
    info_fig4 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig4, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(5)
    plt.title('Figura 5: Funcion Cuadrada de 4KHz')
    plt.grid(True)
    plt.plot(tsq1, xsq1, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    potencia = np.sum(xsq1 ** 2)
    info_fig5 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig5, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(6)
    plt.title('Figura 6: Funcion Cuadrada de 10ms')
    plt.grid(True)
    plt.plot(tsq2, xsq2, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    potencia = np.sum(xsq2 ** 2)
    info_fig6 = (
        f"Tiempo entre muestras: {ts} s\n"
        f"Número de muestras: {N}\n"
        f"Potencia: {potencia} W"
    )
    plt.figtext(0.5, -0.1, info_fig6, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()

#%%############
## Punto (2) ##
###############
#%% Verifico ortogonalidad entre la primera señal y las demás
Xort_12 = np.sum(x1 * x2)
Xort_1_mod = np.sum(x1 * x_modulada)
Xort_1_rec = np.sum(x1 * x_recortada)
Xort_1_sq1 = np.sum(x1 * xsq1)
Xort_1_sq2 = np.sum(x1 * xsq2)

def plotear_tabla_ortogonalidad():
    datos = [["Fig 1 vs Fig 2", Xort_12, "Sí"],
             ["Fig 1 vs Fig 3", Xort_1_mod, "Sí"],
             ["Fig 1 vs Fig 4", Xort_1_rec, "No"],
             ["Fig 1 vs Fig 5", Xort_1_sq1, "Parecerian serlo"],
             ["Fig 1 vs Fig 6", Xort_1_sq2, "No"]]
    print(tabulate(datos, headers=["Figuras a verificar", "Resultado", "¿Son ortogonales?"]))

#%%############
## Punto (3) ##
###############
#%% Grafico la autocorrelación de la primera señal y la correlación entre ésta y las demás.
Xcorr_11 = sig.correlate(x1, x1, mode = 'full')
Xcorr_12 = sig.correlate(x1, x2, mode = 'full')
Xcorr_1_mod = sig.correlate(x1, x_modulada, mode = 'full')
Xcorr_1_rec = sig.correlate(x1, x_recortada, mode = 'full')
Xcorr_1_sq1 = sig.correlate(x1, xsq1, mode = 'full')
Xcorr_1_sq2 = sig.correlate(x1, xsq2, mode = 'full')

def plotear_fig7_a_fig12():
    plt.figure(7)
    plt.title('Figura 7: Autocorrelación de la Fig.1 consigo misma')
    plt.grid(True)
    plt.plot(Xcorr_11, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    
    plt.figure(8)
    plt.title('Figura 8: Autocorrelación de la Fig.1 con la Fig.2')
    plt.grid(True)
    plt.plot(Xcorr_12, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    
    plt.figure(9)
    plt.title('Figura 9: Autocorrelación de la Fig.1 con la Fig.3')
    plt.grid(True)
    plt.plot(Xcorr_1_mod, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    
    plt.figure(10)
    plt.title('Figura 10: Autocorrelación de la Fig.1 con la Fig.4')
    plt.grid(True)
    plt.plot(Xcorr_1_rec, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    
    plt.figure(11)
    plt.title('Figura 11: Autocorrelación de la Fig.1 con la Fig.5')
    plt.grid(True)
    plt.plot(Xcorr_1_sq1, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")
    
    plt.figure(12)
    plt.title('Figura 12: Autocorrelación de la Fig.1 con la Fig.6')
    plt.grid(True)
    plt.plot(Xcorr_1_sq2, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("F(t)")

#%% Mostrar que la igualdad se cumple
frec = 2000
# N = 1000 (definido en linea 24)
nn = np.arange(N, dtype=float)
tt = nn / fs

alfa = frec * tt
beta = alfa/2

termino_izq = 2 * np.sin(alfa) * np.sin(beta)
termino_der = np.cos(alfa - beta) * np.cos(alfa + beta)

def printear_demostracion():
    if ((abs(termino_izq-termino_der) < 1e-9).all): # .all sirve para comparar todos los valores del vector con el escalar 1e-9
        print("Los términos son iguales, por lo que la igualdad se cumple.\n")
    else:
        print("Los terminos no son iguales, por lo que la igualdad no se cumple.\n")
        
#%%############
##   Bonus   ##
###############
def plotear_bonus():
    archivo = 'TS1_Bonus.wav'
    
    fs_wav, x_wav = waves.read(archivo)
    n_wav = len(x_wav)
    ts_wav = 1/fs_wav
    t_wav = np.arange(0, n_wav * ts_wav, ts_wav)
    uncanal = x_wav[:,0]
    t_final_wav = t_wav[-1]
    
    plt.figure(13)
    plt.title('Figura 13: Señal de sonido')
    plt.grid(True)
    plt.plot(t_wav, uncanal, color = 'teal')
    plt.xlim(0, t_final_wav)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Sonido')
    
# %% Ploteo todo

