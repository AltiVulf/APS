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

#%%########################
## Definiciones Globales ##
###########################

fs = 1000
nyquist = fs/2

#%%#########################
## Funciones del Programa ##
############################



#%%##########
# Plantilla #
#############
# %% -- Parámetros de Plantilla de Diseño del filtro --
i=0
wp = [0.8, 35] # frecuencia de corte [Hz]
ws = [0.1, 40] # frecuencia de stop [Hz]

# alpha_pd = 1 # alpha maximo
# alpha_sd = 40 # alpha minimo

# Aproximadores de módulo:
f_aprox = ['butter', 'cheby1', 'cheby2', 'ellip', 'cauer']

numtaps = 2001 # cantidad de coeficientes (PAR)
retardo = (numtaps - 1)//2 # o demora

numfreqs = int((np.ceil(np.sqrt(numtaps))*2)**2 - 1)
freqs = np.sort( np.concatenate(( (0,nyquist),wp,ws )) )          
deseado = [0,0,1,1,0,0] # respuesta deseada (la elijo yo)
# pesos = 

# luego probar sig.remez
fir_win_box = sig.firwin2(numtaps = numtaps, freq = freqs, nfreqs = numfreqs, gain = deseado, fs = fs, window = 'boxcar')
fir_win_box = np.convolve(fir_win_box, fir_win_box)

w, h = sig.freqz(b = fir_win_box, worN = np.logspace(-2,2,1000), fs = fs)

phase = np.unwrap(np.angle(h)) # fase del grupo

w_rad = w / ((fs/2) * np.pi)
gd = -np.diff(phase) / np.diff(w_rad) # retardo

# %%  -- Calculo polos y ceros --
# z, p, k = sig.sos2zpk(sig.tf2sos(b=fir_win_hamming, a=1)) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]

# %% Grafico NOTA: armar función para plotear con una función "ON/OFF" para imprimir (o no) los polos y ceros
# nota 2: poner todo en subplots
# Magnitud
plt.figure(1)
plt.plot(w, 20*np.log10(abs(h)), label = f"{f_aprox[i]}")
# plt.plot(w, 20*np.log10(abs(h_dig)), label = f"{f_aprox[i]}")
plt.title('Figura 1: Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
# plt.xlim(0.1, 10)
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.figure(2)
plt.plot(w, np.degrees(phase), label = f"{f_aprox[i]}")
# plt.plot(w, np.degrees(phase), label = f"{f_aprox[i]}")
plt.title('Figura 2: Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [°]')
# plt.xlim(0.1, 10)
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.figure(3)
plt.plot(w[:-1], gd, label = f"{f_aprox[i]}")
plt.title('Figura 3: Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [s]')
# plt.xlim(0.1, 10)
plt.grid(True, which='both', ls=':')
plt.legend()

# # Diagrama de polos y ceros
# plt.figure(4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label= f'Polos de {f_aprox[i]}')
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'Ceros de {f_aprox[i]}')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Figura 4: Diagrama de Polos y Ceros (plano Z)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

ecg_filt_win = sig.lfilter(b = fir_win_box, a = 1, x = ecg_one_lead)

#%%##############################
# Regiones de interés sin ruido #
#################################

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, fir_win_box[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#%%##############################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, fir_win_box[zoom_region], label='Box')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

#%%############
## Punto (1) ##
###############
