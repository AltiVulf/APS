"""
Created on Wed Aug 28 2025

@author: Tomás Altimare Bercovich

Descripción:
------------
    En este trabajo práctico se realizó el análisis 
    de diferentes sisemas LTI excitados con señales 
    realizadas en el trabajo semanal anterior con un 
    generador de señales, graficando las respuestas 
    resultantes a dichas entradas. Se encontró la 
    respuesta al impulso de los sistemas planteados 
    a partir de la función Delta de Dirac utilizando 
    propiedades de los sistemas LTI. Por último, se 
    intentó discretizar la ecuación diferencial 
    correspondiente al modelo Windkessel que describe 
    la dinámica presión-flujo del sistema sanguíneo.

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
import TS1_Sintesis_de_Señales as TS1 # Importo el TS1 para poder usar las señales

#%%########################
## Definiciones Globales ##
###########################
fs = 1000 # Frecuencia de sampleo
N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
T_simulacion = N/fs # Tiempo total de simulación
n = np.arange(N) # Las N muestras equiespaciadas
tt = n/fs # Grilla temporal de sampleo

#%% Defino la función Delta
def delta():
    x = np.zeros(N, dtype = float)
    x[1] = 1
    return x

#%%############
## Punto (1) ##
###############

#%% Declaro la función y[n]
def ec_LTI_punto_1(x, N):
    # x: Señal de entrada, N: numero de muestras
    
    """
        Nota: Armo los if con el objetivo de que,
        cuando n sea 0 ó 1, y x[n]=x[-1] ó x[-2],
        Python no tome los últimos valores del array.
        Esto lo hago debido a que quiero que
        x[-1] = x[-2] = y[-1] = y[-2] = 0, pero a su
        vez, no quiero modificar los últimos valores
        de los respectivos array.
    """
    
    y = np.zeros(N, dtype = float)
    for n in range(N):
        if n == 0: 
            y[n] = (3*(10**(-2))*x[n])
        elif n == 1:
            y[n] = (3*(10**(-2))*x[n] + 
                    5*(10**(-2))*x[n-1])
        elif n > 1:
            y[n] = (3*(10**(-2))*x[n] + 
                    5*(10**(-2))*x[n-1] + 
                    3*(10**(-2))*x[n-2] +
                    1.5*y[n-1]-0.5*y[n-2])
    return y

#%% Sintetizo las respuestas a las diferentes señales de entrada
respuesta_señal_f1 = ec_LTI_punto_1(TS1.x1, N)
respuesta_señal_f2 = ec_LTI_punto_1(TS1.x2, N)
respuesta_señal_f3 = ec_LTI_punto_1(TS1.x_modulada, N)
respuesta_señal_f4 = ec_LTI_punto_1(TS1.x_recortada, N)
respuesta_señal_f5 = ec_LTI_punto_1(TS1.xsq1, N)
respuesta_señal_f6 = ec_LTI_punto_1(TS1.xsq2, N)

#%% Imprimo las respuestas
def plotear_fig1_a_fig6 ():
    plt.figure(1)
    plt.title('Figura 1: Respuesta al seno con frec = 2KHz')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f1, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    # Añado una leyenda con los datos sobre la señal:
    potencia = np.sum(respuesta_señal_f1 ** 2)/(2*N+1)
    info_fig1 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Potencia: {potencia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig1, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(2)
    plt.title('Figura 2: Respuesta al seno con frec = 2KHz desfasada π/2')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f2, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    potencia = np.sum(respuesta_señal_f2 ** 2)/(2*N+1)
    info_fig2 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Potencia: {potencia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig2, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(3)
    plt.title('Figura 3: Respuesta al seno de 2KHz modulada con Seno de 1KHz')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f3, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    potencia = np.sum(respuesta_señal_f3 ** 2)/(2*N+1)
    info_fig3 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Potencia: {potencia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig3, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(4)
    plt.title('Figura 4: Respuesta al seno de 2KHz recortado al 75%')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f4, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    potencia = np.sum(respuesta_señal_f4 ** 2)/(2*N+1)
    info_fig4 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Potencia: {potencia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig4, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(5)
    plt.title('Figura 5: Respuesta a la funcion cuadrada de 4KHz')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f5, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    potencia = np.sum(respuesta_señal_f5 ** 2)
    info_fig5 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Potencia: {potencia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig5, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(6)
    plt.title('Figura 6: Respuesta a la funcion cuadrada de 10ms')
    plt.grid(True)
    plt.plot(tt, respuesta_señal_f6, 'o--', color = 'teal')
    plt.xlim(0, T_simulacion)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    energia = np.sum(respuesta_señal_f6 ** 2)
    info_fig6 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s\n"
        f"Energía: {energia:.3f}"
    )
    plt.figtext(0.5, -0.1, info_fig6, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()

#%% Busco la respuesta al impulso: Para ello, busco excitar a mi sistema LTI con una delta
h_punto1 = ec_LTI_punto_1(delta(), N)
respuesta_señal_f1_con_h_punto1 = np.convolve(TS1.x1, h_punto1, mode='full')
respuesta_señal_f2_con_h_punto1 = np.convolve(TS1.x2, h_punto1, mode='full')
respuesta_señal_f3_con_h_punto1 = np.convolve(TS1.x_modulada, h_punto1, mode='full')
respuesta_señal_f4_con_h_punto1 = np.convolve(TS1.x_recortada, h_punto1, mode='full')
respuesta_señal_f5_con_h_punto1 = np.convolve(TS1.xsq1, h_punto1, mode='full')
respuesta_señal_f6_con_h_punto1 = np.convolve(TS1.xsq2, h_punto1, mode='full')

#%% Imprimo la respuesta del sistema tras convolucionar la señal f1 con con la respuesta al impulso del sistema

def plotear_fig7_a_fig12():
    # plt.figure(7)
    # plt.title('Figura 7: Respuesta de f1 a partir de convolución con h (Punto 1)')
    # plt.grid(True)
    # plt.plot(respuesta_señal_f1_con_h_punto1, 'o--', color = 'teal')
    # plt.xlim(0, 2000)
    # plt.xlabel("Tiempo [s]")
    # plt.ylabel("Amplitud")
    # # Añado una leyenda con los datos sobre la señal:
    # potencia = np.sum(respuesta_señal_f1_con_h_punto1 ** 2)/(2*N+1)
    # info_fig1 = (
    #     f"Frecuencia de muestreo: {fs} s\n"
    #     f"Tiempo entre muestras: {ts} s\n"
    #     f"Potencia: {potencia:.3f}"
    # )
    # plt.figtext(0.5, -0.1, info_fig1, fontsize=12, ha="center", va="center",
    #             bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    # plt.tight_layout()
    
    figuras = [
        ("Figura 7: Respuesta de f1 a partir de convolución con h (Punto 1)", respuesta_señal_f1_con_h_punto1),
        ("Figura 8: Respuesta de f2 a partir de convolución con h (Punto 1)", respuesta_señal_f2_con_h_punto1),
        ("Figura 9: Respuesta de f3 a partir de convolución con h (Punto 1)", respuesta_señal_f3_con_h_punto1),
        ("Figura 10: Respuesta de f4 a partir de convolución con h (Punto 1)", respuesta_señal_f4_con_h_punto1),
        ("Figura 11: Respuesta de f5 a partir de convolución con h (Punto 1)", respuesta_señal_f5_con_h_punto1),

    ]
    
    for i, (titulo, data) in enumerate(figuras, start=7):
        plt.figure(i)
        figura = i-6
        plt.title(f"Figura {i}: Respuesta de f{figura} a partir de convolución con h (Punto 1)")
        plt.grid(True)
        plt.plot(data, 'o--', color='teal')
        plt.xlim(0, 2000)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        # Añado una leyenda con los datos sobre la señal:
        potencia = np.sum(data ** 2)/(2*N+1)
        info_fig1 = (
            f"Frecuencia de muestreo: {fs} s\n"
            f"Tiempo entre muestras: {ts} s\n"
            f"Potencia: {potencia:.3f}"
        )
        plt.figtext(0.5, -0.1, info_fig1, fontsize=12, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
        plt.tight_layout()
    
        plt.figure(12)
        plt.title('Figura 12: Respuesta de f6 a partir de convolución con h (Punto 1)')
        plt.grid(True)
        plt.plot(respuesta_señal_f6_con_h_punto1, 'o--', color = 'teal')
        plt.xlim(0, 2000)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        # Añado una leyenda con los datos sobre la señal:
        energia = np.sum(respuesta_señal_f6_con_h_punto1 ** 2)/(2*N+1)
        info_fig1 = (
            f"Frecuencia de muestreo: {fs} s\n"
            f"Tiempo entre muestras: {ts} s\n"
            f"Energia: {energia:.3f}"
        )
        plt.figtext(0.5, -0.1, info_fig1, fontsize=12, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
        plt.tight_layout()
        
#%%############
## Punto (2) ##
###############
#%% Declaro las funciones y[n]
def ec1_LTI_punto_2(x, N):
    # x: Señal de entrada, N: numero de muestras
    
    y = np.zeros(N, dtype = float)
    for n in range(N):
        if n < 10:
            y[n] = x[n]
        else:
            y[n] = x[n] + 3*x[n-10]
    return y

def ec2_LTI_punto_2(x, N):
    # x: Señal de entrada, N: numero de muestras
    
    y = np.zeros(N, dtype = float)
    for n in range(N):
        for n in range(N):
            if n < 10:
                y[n] = x[n]
            else:
                y[n] = x[n] + 3*y[n-10]
    return y

#%% Hallar la respuesta al impulso y la salida correspondiente a una señal de entrada senoidal
h1_punto2 = ec1_LTI_punto_2(delta(), N)
h2_punto2 = ec2_LTI_punto_2(delta(), N)

respuesta_señal_f1_con_h1_punto2 = np.convolve(TS1.x1, h1_punto2, mode='full')
respuesta_señal_f1_con_h2_punto2 = np.convolve(TS1.x1, h2_punto2, mode='full')

def plotear_fig13_a_fig14():
    plt.figure(13)
    plt.title('Figura 13: Respuesta de f1 a partir de convolución con h1 (Punto 2)')
    plt.grid(True)
    plt.plot(respuesta_señal_f1_con_h1_punto2, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    # Añado una leyenda con los datos sobre la señal:
    info_fig1 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s"
    )
    plt.figtext(0.5, -0.05, info_fig1, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(14)
    plt.title('Figura 14: Respuesta de f1 a partir de convolución con h2 (Punto 2)')
    plt.grid(True)
    plt.plot(respuesta_señal_f1_con_h2_punto2, 'o--', color = 'teal')
    plt.xlim(0, 2000)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    # Añado una leyenda con los datos sobre la señal:
    info_fig1 = (
        f"Frecuencia de muestreo: {fs} s\n"
        f"Tiempo entre muestras: {ts} s"
    )
    plt.figtext(0.5, -0.05, info_fig1, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
#%% Ploteo de prueba
plotear_fig1_a_fig6()
plotear_fig7_a_fig12()
plotear_fig13_a_fig14()

#%%##############
## Punto Bonus ##
#################
#%% Defino constantes de Compliance y Resistencia vascular
def plotear_bonus():
    t_sim = 5
    
    compilance = 0.00648 # cm³/mmHg
    res_vascular = 81.68 # mmHg/cm³/sec
    t = np.arange(0, t_sim) # Eje del tiempo
    Pi = 80
    
    P = Pi * np.exp(-(t-t[0]) / (res_vascular*compilance)) # Ecuación de la presión con respecto del tiempo
    Pdt = P * (t-t[0]) / ((res_vascular * compilance)**2)
    
    Q = compilance * Pdt + P/res_vascular
    
    plt.figure(10)
    plt.title('Figura 10: Flujo de sangre en el tiempo')
    plt.grid(True)
    plt.plot(Q, 'x--', color = 'blue')
    plt.plot(P, 'x--', color = 'red')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Flujo")
    
    plt.tight_layout()