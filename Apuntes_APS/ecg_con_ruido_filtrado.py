#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:36:45 2025

@author: venta
"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.fft import fft, fftshift
from scipy import signal as sig 
import scipy.io as sio
import scipy.signal.windows as window
from scipy.signal import periodogram
from scipy.io.wavfile import write

#%%########################
## Definiciones Globales ##
###########################

fs = 1000 # Frecuencia de Sampleo
# N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
# n = np.arange(N) # Las N muestras equiespaciadas

#%%#########################
## Funciones del Programa ##
############################


#%%################
## ECG con ruido ##
###################

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_con_ruido = mat_struct['ecg_lead'].flatten()
N = len(ecg_con_ruido)

# %% -- Parámetros de Plantilla de Diseño del filtro --
fpaso = [0.8, 35] # frecuencia de corte [Hz]
fstop = [0.1, 40] # frecuencia de stop [Hz]

alpha_pd = 1 # alpha maximo
alpha_sd = 40 # alpha minimo

f_aprox = ['butter', 'cheby1', 'cheby2', 'ellip', 'cauer']

sos_butt = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                     analog = False, ftype = f_aprox[0], output = 'sos', fs = fs)

sos_cheby1 = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                     analog = False, ftype = f_aprox[1], output = 'sos', fs = fs)

sos_cheby2 = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                     analog = False, ftype = f_aprox[2], output = 'sos', fs = fs)

sos_ellip = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                     analog = False, ftype = f_aprox[3], output = 'sos', fs = fs)

sos_cauer = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                     analog = False, ftype = f_aprox[4], output = 'sos', fs = fs)

ecg_filt_butt = sig.sosfilt(sos_butt, ecg_con_ruido)
ecg_filt_cheby1 = sig.sosfilt(sos_cheby1, ecg_con_ruido)
ecg_filt_cheby2 = sig.sosfilt(sos_cheby2, ecg_con_ruido)
ecg_filt_ellip = sig.sosfilt(sos_ellip, ecg_con_ruido)
ecg_filt_cauer = sig.sosfilt(sos_cauer, ecg_con_ruido)

ecg_2filt_butt = sig.sosfiltfilt(sos_butt, ecg_con_ruido)
ecg_2filt_cheby1 = sig.sosfiltfilt(sos_cheby1, ecg_con_ruido)
ecg_2filt_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_con_ruido)
ecg_2filt_ellip = sig.sosfiltfilt(sos_ellip, ecg_con_ruido)
ecg_2filt_cauer = sig.sosfiltfilt(sos_cauer, ecg_con_ruido)

# %% Filtro el ECG
plt.figure(1)
plt.clf()
plt.title('ECG Filtrado')
plt.plot(ecg_con_ruido[:50000], label = 'ECG original', color = 'blue')
plt.plot(ecg_filt_butt[:50000], label = 'butter')
plt.plot(ecg_filt_cheby1[:50000], label = 'cheby1')
plt.plot(ecg_filt_cheby2[:50000], label = 'cheby2')
plt.plot(ecg_filt_ellip[:50000], label = 'ellip')
plt.plot(ecg_filt_cauer[:50000], label = 'cauer')
plt.xlabel('Muestras [N]')
plt.ylabel('Amplitud')
plt.legend()

# %% Meto el ECG dos veces en el filtro para sacar TODA la distorsion de fase 
plt.figure(2)
plt.clf()
plt.title('ECG Doble Filtrado - Filtrado Bidireccional')
plt.plot(ecg_con_ruido[:50000], label = 'ECG original', color = 'blue')
plt.plot(ecg_2filt_butt[:50000], label = 'butter')
plt.plot(ecg_2filt_cheby1[:50000], label = 'cheby1')
plt.plot(ecg_2filt_cheby2[:50000], label = 'cheby2')
plt.plot(ecg_2filt_ellip[:50000], label = 'ellip')
plt.plot(ecg_2filt_cauer[:50000], label = 'cauer')
plt.xlabel('Muestras [N]')
plt.ylabel('Amplitud')
plt.legend()
# Aparece la respuesta a la impulso para un lado y para el otro.
# Neutralizo/anulo la respuesta fase --> Cancelo la demora
# 
# %% Imprimo magnitud, fase, retardo de grupo, polos y ceros.
for i in range(0, 5):
    dig_sos = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                         analog = False, ftype = f_aprox[i], output = 'sos', fs = fs)
    # Respuesta en frecuencia del filtro (calculada) 
    w_dig, h_dig = sig.freqz_sos(dig_sos, fs = fs) # omega en Hz
    phase = np.unwrap(np.angle(h_dig)) # fase del grupo
    
    w_dig_rad = w_dig / ((fs/2) * np.pi)
    gd = -np.diff(phase) / np.diff(w_dig_rad) # retardo

    # %%  -- Calculo polos y ceros --
    z_dig, p_dig, k_dig = sig.sos2zpk(dig_sos) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]
    
    # %% Grafico
    # Magnitud
    plt.figure(3)
    plt.semilogx(w_dig, 20*np.log10(abs(h_dig)), label = f"{f_aprox[i]}")
    # plt.plot(w_dig, 20*np.log10(abs(h_dig)), label = f"{f_aprox[i]}")
    plt.title('Figura 1: Respuesta en Magnitud (o de Módulo)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Fase
    plt.figure(4)
    plt.semilogx(w_dig, np.degrees(phase), label = f"{f_aprox[i]}")
    # plt.plot(w_dig, np.degrees(phase), label = f"{f_aprox[i]}")
    plt.title('Figura 2: Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [°]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Retardo de grupo
    plt.figure(5)
    plt.semilogx(w_dig[:-1], gd, label = f"{f_aprox[i]}")
    plt.title('Figura 3: Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [s]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Diagrama de polos y ceros
    plt.figure(6) # ver de imprimirlos todos por separado
    plt.plot(np.real(p_dig), np.imag(p_dig), 'x', markersize=10, label= f'Polos de {f_aprox[i]}')
    if len(z_dig) > 0:
        plt.plot(np.real(z_dig), np.imag(z_dig), 'o', markersize=10, fillstyle='none', label=f'Ceros de {f_aprox[i]}')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.title('Figura 4: Diagrama de Polos y Ceros (plano Z)')
    plt.xlabel('σ [rad/s]')
    plt.ylabel('jω [rad/s]')
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    unit_circle = patches.Circle((0,0), radius = 1, fill = False,
                                 color = 'gray', ls = 'dotted', lw = 2)
    axes_hdl = plt.gca()
    axes_hdl.add_patch(unit_circle)
    plt.legend()
    plt.grid(True)
    
# tarea: revisar efectividad en alta frecuencia

# Respuesta al impulso de un filtro FIR --> de fase lineal
