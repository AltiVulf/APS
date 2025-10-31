"""
Created on Wed Aug 20 2025

@author: Tomás Altimare Bercovich

Descripción:
------------
    Ploteo de tres funciones transferencia:
    Módulo, Fase, Retardo de Fase, Polos y Ceros.

    El programa contiene una función armada donde se
    plotean los aspectos antes mencionados.
    
"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import signal as sig 

#%%#########################
## Funciones del Programa ##
############################
def plotear_funcion_transferencia(a, b, xlim = (0, 10), label = 'T(s)', xlim_s = False, polos_y_ceros = True, magnitud_fase_retardo = True, fig_inicial = 0):
    # --- Calculo polos y ceros ---
    z, p, k = sig.tf2zpk(b, a) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]

    # --- Módulo, Fase y Retardo de Fase ---
    w, h = sig.freqs(b, a) # worN = np.logspace(2,-2,1000)
    phase = np.unwrap(np.angle(h)) # fase del grupo
    gd = -np.diff(phase) / np.diff(w) # retardo

    # --- Ploteo ---
    figura = 1
    if magnitud_fase_retardo == True:
        # Magnitud
        plt.figure(fig_inicial + figura)
        plt.title(f'Figura {fig_inicial + figura}: Respuesta en Magnitud')
        plt.xlabel('Pulsación angular [r/s]')
        plt.ylabel('|H(jω)| [dB]')
        plt.plot(w, 20*np.log10(abs(h)), label = f"{label}")
        if xlim_s == True: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend()
        figura+=1
    
        # Fase
        plt.figure(fig_inicial + figura)
        plt.title(f'Figura {fig_inicial + figura}: Respuesta de Fase')
        plt.xlabel('Pulsación angular [r/s]')
        plt.ylabel('Fase [°]')
        plt.plot(w, np.degrees(phase), label = f"{label}")
        if xlim_s == True: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend()
        figura+=1
        
        # Retardo de grupo
        plt.figure(fig_inicial + figura)
        plt.title(f'Figura {fig_inicial + figura}: Retardo de Grupo')
        plt.xlabel('Pulsación angular [r/s]')
        plt.plot(w[:-1], gd, label = f"{label}")
        plt.ylabel('τg [s]')
        if xlim_s == True: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend()
        figura+=1

    # Diagrama de polos y ceros
    if polos_y_ceros == True:
        plt.figure(fig_inicial + figura, figsize = (5,5))
        plt.title(f'Figura {fig_inicial + figura}: Diagrama de Polos y Ceros de {label} (plano s)')
        plt.xlabel('σ [rad/s]')
        plt.ylabel('jω [rad/s]')
        plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label= f'Polos de {label}')
        if len(z) > 0:
            plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'Ceros de {label}')
        plt.axhline(0, color='k', lw=0.5)
        plt.axvline(0, color='k', lw=0.5)
        # Grafico el circulo unitario
        unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                             color='gray', ls='dotted', lw=2)
        axes_hdl = plt.gca()
        axes_hdl.add_patch(unit_circle)

        plt.axis([-1.1, 1.1, -1.1, 1.1])
        plt.legend()
        plt.grid(True)

#%%############
## Punto (3) ##
###############
# %% Armo las funciones Transferencia
b1 = [1, 0, 9]
a1 = [1, np.sqrt(2), 1]

b2 = [1, 0, (1/9)]
a2 = [1, (1/5), 1]

b3 = [1, (1/5), 1]
a3 = [1, np.sqrt(2), 1]

# %% Ploteo los polos y ceros
plotear_funcion_transferencia(a1, b1, label = 'T1', magnitud_fase_retardo = False)
plotear_funcion_transferencia(a2, b2, label = 'T2', magnitud_fase_retardo = False, fig_inicial = 1)
plotear_funcion_transferencia(a3, b3, label = 'T3', magnitud_fase_retardo = False, fig_inicial = 2)

# %% Ploteo magnitud, fase y retardo de grupo
plotear_funcion_transferencia(a1, b1, label = 'T1', xlim = (0, 10), xlim_s = True, polos_y_ceros = False, fig_inicial = 3)
plotear_funcion_transferencia(a2, b2, label = 'T2', xlim = (0, 10), xlim_s = True, polos_y_ceros = False, fig_inicial = 3)
plotear_funcion_transferencia(a3, b3, label = 'T3', xlim = (0, 10), xlim_s = True, polos_y_ceros = False, fig_inicial = 3)