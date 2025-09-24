"""
Created on Wed Aug 27 2025

@author: Tomás Altimare Bercovich

Descripción:
------------


"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as window
from numpy.fft import fft, fftshift

#%% Defino las variables que voy a usar
N = 1000 # Cantidad de muestras
fs = 1000 # Frecuencia de sampleo
df = fs/N # Resolución espectral
ts = 1/fs # Tiempo/Periodo de sampleo
T_simulacion = N/fs # Duración total de la simulación
ff = np.arange(N) * df # Vector en frecuencia al escalar las muestras por la resolución espectral

#%% Defino el seno
def funcion_senoidal(ff, nn, vmax = 1, dc = 0, ph = 0, fs = 2): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje vertical), funcion (eje horizontal)
    """    
    # grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

#%% Defino una función para imprimir en dB
def dB(W):
    return 20 * np.log10(np.abs(np.fft.fftshift(W)) / np.max(np.abs(W)))

#%%############
## Punto (1) ##
###############
#%% Parametrizo el seno
Vmax = np.sqrt(2)
dc = 0
frec = (N/4)*df

t1, x1 = funcion_senoidal(ff = frec, nn = N, vmax = Vmax, fs = fs)

# plt.figure(1)
# plt.title('Figura 1: Seno')
# plt.grid(True)
# plt.plot(t1, x1, 'o--', color = 'teal')
# plt.xlim(0, T_simulacion)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("F(t)")

#%% Imprimo los valores
print(f"Media: {np.mean(x1):.5f}")
print(f"Varianza: {np.var(x1):.5f}") # Varianza
print(f"Desviacion Estandar: {np.std(x1):.5f}") # Desviacion estandar

#%%############
## Punto (2) ##
###############
#%% Verifico Parseval
X1_fft = fft(x1)
A = np.sum(np.abs(x1)**2)
B = np.sum(np.abs(X1_fft)**2)/N

parseval = np.abs(A-B)

if (parseval < 10e-9):
    print("\nSe cumple parseval\n")

#%%############
## Punto (3) ##
###############
#%% Cero Padding (forma de interpolar y mejorar la resolucion espectral)
zero = np.zeros(len(x1)*9)
cero_padding = np.concatenate((x1, zero))
cero_padding = fft(cero_padding)

t_padding = np.arange (10*N) * (fs / (10*N))

plt.figure(2)
plt.title('Figura 2: Cero Padding del Seno (Sinc)')
plt.grid(True)
plt.plot(t_padding, 20*np.log10(np.abs(cero_padding)), 'o', color = 'teal')
plt.plot(ff, np.log10(20*np.abs(X1_fft)),'x', color = 'firebrick')
plt.xlim(0, N/2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X1|\u00b2 [dB]") # \u00b2 sirve para poner el 2 al cuadrado
plt.tight_layout()

def zero_padding(sig, N, fs): 
    """
    padding(sig: señal original, N: # muestras, fs: frecuencia de muestreo)
    """
    zeroPadding = np.zeros(N) # Lleno el vector de ceros
    zeroPadding[:len(sig)] = sig # Meto la señal dentro del vector
    
    sig_padding = np.abs(fft(zeroPadding)) # 
    
    return sig_padding

#%%############
## Punto (4) ##
###############
#%% Ventaneo: Empiezo llamando algunas ventanas de scipy
N_HD = N*9 # N 
flattop = window.flattop(N_HD)
blackmanharris = window.blackmanharris(N_HD)
hamming = window.hamming(N_HD)
taylor = window.taylor(N_HD)
bohman = window.bohman(N_HD)
# Armo las FFT de cada una
fft_flattop = zero_padding(sig = flattop, N = N_HD, fs = fs)
fft_blackmanharris = zero_padding(sig = blackmanharris, N = N_HD, fs = fs)
fft_hamming = zero_padding(sig = hamming, N = N_HD, fs = fs)
fft_taylor = zero_padding(sig = taylor, N = N_HD, fs = fs)
fft_bohman = zero_padding(sig = bohman, N = N_HD, fs = fs)

eje_frec = np.linspace(-fs/2, fs/2, len(fft_flattop))

#%% Grafico las ventanas en el tiempo
plt.figure(3)
plt.clf()
plt.title('Figura 3: Sampleo temporal')
plt.grid(True)
plt.plot(flattop, label = 'Flattop', color = 'orangered')
plt.plot(blackmanharris, label = 'Blackman Harris', color = 'forestgreen')
plt.plot(hamming, label = 'Hamming', color = 'cornflowerblue')
plt.plot(taylor, label = 'Taylor',color = 'gold')
plt.plot(bohman, label = 'Bohman',color = 'fuchsia')
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.xlim(0, N_chiquita)
plt.tight_layout()

plt.figure(4)
plt.clf()
plt.title('Figura 4: Respuesta en frecuecia')
plt.grid(True)
plt.plot(eje_frec, dB(fft_flattop), label = 'Flattop', color = 'orangered')
plt.plot(eje_frec, dB(fft_blackmanharris), label = 'Blackman Harris', color = 'forestgreen')
plt.plot(eje_frec, dB(fft_hamming), label = 'Hamming', color = 'cornflowerblue')
plt.plot(eje_frec, dB(fft_taylor), label = 'Taylor', color = 'gold')
plt.plot(eje_frec, dB(fft_bohman), label = 'Bohman', color = 'fuchsia')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|W(k)| [dB]") # \u00b2 sirve para poner el 2 al cuadrado
plt.legend()
plt.xlim(-fs/2, fs/2)
plt.ylim(min(dB(fft_blackmanharris))-5, 5)
plt.tight_layout()
