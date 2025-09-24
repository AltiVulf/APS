
#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import scipy.signal.windows as window

N = 10000

A = np.random.randn(N)
B = np.random.randn(N)

plt.figure(1, figsize=(10,10))
plt.plot (A,B,'x')
plt.plot (A,A,'+')
plt.plot (A,0.7*A + 0.4*B, '.')