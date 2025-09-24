#%% Modulos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Funciones
plt.close("all")

def mi_funcion_sen_estocastica_matricial(vmax, dc, ff, fr_matriz, realizaciones, ph, N, fs, plot=True):
    
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).reshape(-1,1) # grilla de sampleo temporal discreta (n) pasa de vector en 1xN a matriz NX1 (-1 toma el tamaño del último elemento)
    tt_matriz = np.tile(tt, (1, realizaciones))   #1000x200  pr tile replica array en columnas= repeticiones y filas =1 = no repitas
    omega_0=ff
    omega_1=(fs/N)*fr_matriz + omega_0
    arg = 2*np.pi*omega_1*tt_matriz + ph # argumento
    xx_matriz = (vmax*(np.sin(arg)) + dc) # señal
    var_x=np.var(xx_matriz)
    
    print(f'La varianza de la señal sin normalizar es: {var_x}\n')    
    if plot:
        
        #%% Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt_matriz, xx_matriz, label=f"f = {ff} Hz\nN = {N}\nTs = {ts} s\nPotencia = {var_x:.3} W")
        plt.title('Señal: senoidal')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid()
        plt.xlim([tt_matriz.min() - 0.1*(tt_matriz.max()-tt_matriz.min()), tt_matriz.max() + 0.1*(tt_matriz.max()-tt_matriz.min())])
        plt.ylim([xx_matriz.min() - 0.1*(xx_matriz.max()-xx_matriz.min()), xx_matriz.max() + 0.1*(xx_matriz.max()-xx_matriz.min())])
        plt.legend()
        plt.show() 
        
    return tt_matriz,xx_matriz


def mi_funcion_noise_matricial(N,SNR,media_n,realizaciones):
    var_n=10**(-SNR/10)
    med_n=media_n
    # n=np.random.normal(med_n, np.sqrt(var_n),N).reshape(-1,1)
    # n_matriz=np.tile(n, (1, realizaciones))
    n = np.random.normal(med_n, np.sqrt(var_n), (N, realizaciones))  # ruido independiente por realización
    n_matriz = n.reshape(N, realizaciones) # redundante
    var_n=np.var(n)
    print(f'La varianza del ruido es: {var_n}\n')
    
    return n_matriz,var_n


def frecuencia_random_matricial(a,b,realizaciones):
    fr=np.random.uniform(a,b,realizaciones).reshape(1,-1) # 1x200
    fr_matriz=np.tile(fr, (N, 1)) #1000x200
    return fr_matriz

def normalizacion(x):
    media_x=np.mean(x) #media
    desvio_x=np.std(x) #desvio
    xx_norm=(x-media_x)/desvio_x #señal normalizada
    varianza_x=np.var(xx_norm) #varianza
    print(f'La varianza de la señal normalizada es: {varianza_x}\n')
    
    return xx_norm,varianza_x

def mod_y_fase_fft(fft_x):
    fft_x_abs=np.abs(fft_x)
    fft_x_abs_ph=np.max(np.angle(fft_x_abs))
    return fft_x_abs, fft_x_abs_ph

def estimadores(fft_abs_x,df,ff):
    k0= int(np.round(ff*(1/df))) # índice redondeado a entero de la fft que corresponde a ff
    a_estimadas=fft_abs_x[k0,:] # Vector de amplitudes en el índice de ff para c/realización
    index=np.argmax(fft_abs_x,axis=0) # Vector de índices donde fft tiene máximo argumento para c/realización
    f_estimadas=index*df # Conversión de índice a frecuencia
    return f_estimadas,a_estimadas

def estadisticas(x_real,estimacion):
    media=np.mean(estimacion)
    sesgo=media-np.mean(x_real)
    varianza=np.var(estimacion,mean=media)
    return sesgo,varianza

#%% Parámetros
N=1000
M=10*N
fs=1000
df=fs/N
k=np.arange(N)*df
k_M=np.arange(M)*df
ts = 1/fs # tiempo de muestreo
realizaciones=200 # parametro de fr

#%% Invocación de las funciones del punto 1
n_m_1,var_n_m_1=mi_funcion_noise_matricial(N=N,SNR=3,media_n=0,realizaciones=realizaciones)
n_m_2,var_n_m_2=mi_funcion_noise_matricial(N=N,SNR=10,media_n=0,realizaciones=realizaciones)

fr_m=frecuencia_random_matricial(a=-2,b=2,realizaciones=realizaciones)

t_m_1,x_m_1 = mi_funcion_sen_estocastica_matricial(vmax = 2, dc = 0, ff = fs/4, fr_matriz=fr_m, realizaciones=realizaciones, ph=0, N=N,fs=fs,plot=None)

# Normalización de señal
x_m_norm_1,var_m_norm_1=normalizacion(x_m_1)

# Señal normalizada con ruido
xn_m_1=x_m_norm_1+n_m_1

# Varianza de señal normalizada con ruido
var_xn_m_1=np.var(xn_m_1)

# Ventanas
w_bh=sig.windows.blackmanharris(N).reshape(-1,1)
w_bh_m=np.tile(w_bh, (1, realizaciones))

w_hamming=sig.windows.hamming(N).reshape(-1,1)
w_hamming_m=np.tile(w_hamming, (1, realizaciones))

w_flattop=sig.windows.flattop(N).reshape(-1,1)
w_flattop_m=np.tile(w_flattop, (1, realizaciones))

w_rectangular=sig.windows.get_window("boxcar",N).reshape(-1,1)
w_rectangular_m=np.tile(w_rectangular, (1, realizaciones))

# Señal normalizada con ruido ventaneada
xn_m_1_w_bh=xn_m_1*w_bh_m
xn_m_1_w_hamming=xn_m_1*w_hamming_m
xn_m_1_w_flattop=xn_m_1*w_flattop_m
xn_m_1_w_rectangular=xn_m_1*w_rectangular_m

# FFT normalizada por N
fft_xn_m_1_w_bh=np.fft.fft(xn_m_1_w_bh,axis=0)/N 
fft_xn_m_1_w_hamming=np.fft.fft(xn_m_1_w_hamming,axis=0)/N 
fft_xn_m_1_w_flattop=np.fft.fft(xn_m_1_w_flattop,axis=0)/N 
fft_xn_m_1_w_rectangular=np.fft.fft(xn_m_1_w_rectangular,axis=0)/N 

# Módulo y fase de FFT
fft_xn_m_1_w_bh_abs,fft_xn_m_1_w_bh_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_bh)
fft_xn_m_1_w_hamming_abs,fft_xn_m_1_w_hamming_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_hamming)
fft_xn_m_1_w_flattop_abs,fft_xn_m_1_w_flattop_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_flattop)
fft_xn_m_1_w_rectangular_abs,fft_xn_m_1_w_rectangular_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_rectangular)

# Estimadores de amplitud y frecuencia para c/ventana
f_estimado_1_xn_1_w_bh,a_estimado_1_xn_1_w_bh=estimadores(fft_xn_m_1_w_bh_abs,df=df,ff=fs/4)
f_estimado_1_xn_1_w_hamming,a_estimado_1_xn_1_w_hamming=estimadores(fft_xn_m_1_w_hamming_abs,df=df,ff=fs/4)
f_estimado_1_xn_1_w_flattop,a_estimado_1_xn_1_w_flattop=estimadores(fft_xn_m_1_w_flattop_abs,df=df,ff=fs/4)
f_estimado_1_xn_1_w_rectangular,a_estimado_1_xn_1_w_rectangular=estimadores(fft_xn_m_1_w_rectangular_abs,df=df,ff=fs/4)

# Sesgo y varianza
a_real=np.tile(2,(realizaciones,1)) # Amplitud real
f_real=fs/4 + df*fr_m[0,:] # Frecuencia real

# Estadisticas
sesgo_a_bh,var_a_bh=estadisticas(a_real, a_estimado_1_xn_1_w_bh)
sesgo_f_bh,var_f_bh=estadisticas(f_real, f_estimado_1_xn_1_w_bh)

print("Ventana Blackman-Harris:")
print(f" Amplitud -> sesgo={sesgo_a_bh:.3f}, var={var_a_bh:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_bh:.3f}, var={var_f_bh:.3e}")

sesgo_a_hamming,var_a_hamming=estadisticas(a_real, a_estimado_1_xn_1_w_hamming)
sesgo_f_hamming,var_f_hamming=estadisticas(f_real, f_estimado_1_xn_1_w_hamming)

print("Ventana Hamming:")
print(f" Amplitud -> sesgo={sesgo_a_hamming:.3f}, var={var_a_hamming:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_hamming:.3f}, var={var_f_hamming:.3e}")

sesgo_a_flattop,var_a_flattop=estadisticas(a_real, a_estimado_1_xn_1_w_flattop)
sesgo_f_flattop,var_f_flattop=estadisticas(f_real, f_estimado_1_xn_1_w_flattop)

print("Ventana Flattop:")
print(f" Amplitud -> sesgo={sesgo_a_flattop:.3f}, var={var_a_flattop:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_flattop:.3f}, var={var_f_flattop:.3e}")

sesgo_a_rectangular,var_a_rectangular=estadisticas(a_real, a_estimado_1_xn_1_w_rectangular)
sesgo_f_rectangular,var_f_rectangular=estadisticas(f_real, f_estimado_1_xn_1_w_rectangular)

print("Ventana Rectangular:")
print(f" Amplitud -> sesgo={sesgo_a_rectangular:.3f}, var={var_a_rectangular:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_rectangular:.3f}, var={var_f_rectangular:.3e}")