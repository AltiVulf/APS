"""
Created on Wed Aug 20 2025

@author: Tomás Altimare Bercovich

Descripción:
------------

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy import signal as sig 
import scipy.io as sio
import scipy.signal.windows as window
from scipy.signal import periodogram
from scipy.io.wavfile import write
# import sounddevice as sd # Sirve para escuchar archivos de audio
# import lectura_sigs as lsigs # Importo las señales de ECG, PPG y .WAV

#%%########################
## Definiciones Globales ##
###########################

fs = 1000 # Frecuencia de sampleo
N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
T_simulacion = N/fs # Tiempo total de simulación
n = np.arange(N) # Las N muestras equiespaciadas

#%%#########################
## Funciones del Programa ##
############################

"""Función para imprimir en dB"""
def dB(X):  
    X_shift = fftshift(X) # Armo la FFT centrada
    X_abs = np.abs(X_shift) # Armo el módulo de la FFT (PSD)(la densidad espectral)
    X_max = np.max(X_abs) if np.max(X_abs)>0 else 1 # Calculo el máximo de la funcion para poder trasladar las funciones al mismo eje
    return 20 * np.log10(X_abs / X_max)

"""Función para hacer Zero Padding"""
def zero_padding(sig, n_zeros = 9): 
    Nsig = len(sig)
    padded = np.concatenate([sig, np.zeros(n_zeros * Nsig)]) # Relleno padded con n cantidad de ceros
    return fft(padded) # Armo la densidad espectral con la fft

def estimar_BW(PSD, ff, cota = 0.98):
    energia_acumulada = np.cumsum(PSD) # El ultimo valor del vector contiene la suma de todos los anteriores (integral de toda la curva)
    energia_acumulada_normalizada = energia_acumulada / energia_acumulada[-1] # De tamaño (nperseg, 1)
    corte = energia_acumulada_normalizada[-1] * cota
    elementos_discriminados = int (np.where(energia_acumulada_normalizada >= corte)[0][0]) # cota es el porcentaje que determina los valores que me quiero quedar de la señal (ej.: cota = 0.99)
    frec_BW = ff[elementos_discriminados]
    return frec_BW
    

#%%############
## Punto (1) ##
###############
# %% Definiciones previas
flattop = window.flattop(N)

# %% ECG sin Ruido
def plot_ECG_sin_ruido():
    ECG_sin_ruido = np.load('ecg_sin_ruido.npy')
    
    cant_promedio = 20 # Cada cuantos sampleos promedio (a mayor valor, menor resolución)
    nperseg = ECG_sin_ruido.shape[0] // cant_promedio
    zpadding = nperseg * 5
    
    ff_P_ECG_SR, P_ECG_SR = periodogram(ECG_sin_ruido, fs = fs, nfft = zpadding)

    ff_SR, Welch_ECG_SR = sig.welch(ECG_sin_ruido, fs = fs, window = 'hann', nperseg = nperseg)
    ff_ZP_SR, ZP_Welch_ECG_SR = sig.welch(ECG_sin_ruido, fs = fs, window = 'hann', nperseg = nperseg, nfft = zpadding)
    
    BW = estimar_BW(ZP_Welch_ECG_SR, ff_ZP_SR, cota = 0.98)
    print(f'\nAncho de Banda ECG sin Ruido: {BW}')
    
    plt.figure(1)
    plt.title('Figura 1: ECG sin ruido')
    plt.grid(True)
    plt.plot(ECG_sin_ruido, color = 'teal', label = 'ECG sin ruido')
    plt.xlim(0, len(ECG_sin_ruido))
    plt.xlabel("Sampleos")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(2)
    plt.title('Figura 2: ECG sin ruido - Periodograma')
    plt.grid(True)
    plt.plot(ff_P_ECG_SR, P_ECG_SR, color = 'teal', label = 'ECG sin ruido - Periodograma')
    plt.xlim(0, 40)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(3)
    plt.title('Figura 3: ECG sin ruido - Método de Welch')
    plt.grid(True)
    plt.plot(ff_SR, Welch_ECG_SR, color = 'teal', label = 'ECG sin ruido')
    plt.xlim(0, 40)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(4)
    plt.title('Figura 4: ECG sin ruido con Zero Padding - Método de Welch')
    plt.grid(True)
    plt.plot(ff_ZP_SR, ZP_Welch_ECG_SR, color = 'teal', label = 'ECG sin ruido')
    plt.xlim(0, 40)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
plot_ECG_sin_ruido()

# %% ECG con ruido
mat_struct = sio.loadmat('./ECG_TP4.mat')
ECG_con_ruido1 = mat_struct['ecg_lead']

# %% PPG sin ruido
def plot_PPG_sin_ruido():
    fs_PPG = 400
    
    PPG = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

    cant_promedio = 40 # Cada cuantos sampleos promedio (a mayor valor, menor resolución)
    nperseg = PPG.shape[0] // cant_promedio
    zpadding = nperseg * 5
    
    ff_P_PPG, P_PPG = periodogram(PPG, fs = fs_PPG, nfft = zpadding)

    ff_PPG, Welch_PPG = sig.welch(PPG, fs = fs_PPG, window = 'hann', nperseg = nperseg)
    ff_ZP_PPG, ZP_Welch_PPG = sig.welch(PPG, fs = fs_PPG, window = 'hann', nperseg = nperseg, nfft = zpadding)
    
    BW = estimar_BW(ZP_Welch_PPG, ff_ZP_PPG, cota = 0.98)
    print(f'\nAncho de Banda PPG sin Ruido: {BW}')
    
    plt.figure(5)
    plt.title('Figura 5: PPG sin ruido')
    plt.grid(True)
    plt.plot(PPG, color = 'teal', label = 'PPG sin ruido')
    plt.xlim(0, len(PPG))
    plt.xlabel("Sampleos")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(6)
    plt.title('Figura 6: PPG sin ruido - Periodograma')
    plt.grid(True)
    plt.plot(ff_P_PPG, P_PPG, color = 'teal', label = 'PPG sin ruido - Periodograma')
    plt.xlim(0, 1)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(7)
    plt.title('Figura 7: PPG sin ruido - Método de Welch')
    plt.grid(True)
    plt.plot(ff_PPG, Welch_PPG, color = 'teal', label = 'PPG sin ruido')
    plt.xlim(0, 5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(8)
    plt.title('Figura 8: PPG sin ruido con Zero Padding - Método de Welch')
    plt.grid(True)
    plt.plot(ff_ZP_PPG, ZP_Welch_PPG, color = 'teal', label = 'PPG sin ruido')
    plt.xlim(0, 5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    plt.tight_layout()
    
plot_PPG_sin_ruido()