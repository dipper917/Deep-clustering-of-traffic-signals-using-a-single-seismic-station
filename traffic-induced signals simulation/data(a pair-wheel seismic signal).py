import math
import cmath
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import interpolate
import matplotlib.pyplot as plt
from SeismicCurve import SeismicCurve
from col2row import col2row
import random
import os

# Basic parameter settings
Fs = 50  # Sampling frequency (Hz)
dt = 0.01  # Sampling interval (s)
A = 1e5  # Amplitude
ns = 1  # Number of sources
nr = 1  # Number of receivers
rx = 0  # Receiver x-coordinate
ry = 0  # Receiver y-coordinate

D = np.zeros([ns, nr])  # Source-receiver distance

# Load fundamental mode velocity
f = open('Rayleigh.txt', 'r', encoding='utf-8')
V = []
while True:
    buf = f.readline()
    if buf:
        index = buf.rfind(' ')
        V.append(eval(buf[index:]))
    else:
        break
f.close()
del V[0]
V = np.array(V)
nn = math.floor(5.12 / dt)
f1 = interpolate.interp1d(np.linspace(0, Fs, 50), V, 'linear')
ff = np.linspace(0, Fs, nn)
V0 = f1(ff)
aa = 0
for i in range(5000):
    fm = random.randint(5,25)

    # Ricker wavelet function
    def Ricker(f1, dt):
        ft = []
        nw = 2.2 / f1 / dt
        nw = 2 * math.floor(nw / 2) + 1
        nc = math.floor(nw / 2)
        w = np.zeros((nw, 1))
        k = np.arange(1, nw + 1, 1)
        k = k.reshape(np.size(k), 1)
        alpha = (-k + nc + 1) * f1 * dt * math.pi
        beta = alpha ** 2
        w = A * (-beta * 2 + 1) * np.exp(-beta)
        # tw = -(nc + 1 - k.T)* dt
        return w.T

    # Source function for a single axle
    wavelet = Ricker(fm, dt)
    source = wavelet
    waveletnt = np.size(wavelet)

    source = np.append(source, np.zeros([1, nn-np.size(source)]))
    H = fft(source)
    nf = np.size(source)
    f = 1 / dt / nf * np.arange(0, nf, 1)

    # Training dataset
    sx = 5
    sy = 0
    D = np.zeros([ns, nr])  # Source-receiver distance
    ts = np.zeros([1, ns])  # Source time delay
    for i in range(ns):
        ts[0, i] = 4.5 * random.random()
    for i in range(nr):
        for k in range(ns):
            D[k, i] = ((rx - sx) ** 2 + (ry - sy) ** 2) ** 0.5

    # Simulation
    U = np.zeros((nn, nr), dtype=complex)
    nfmodel = np.size(V)
    tempdispCF = np.append(np.arange(1, nfmodel + 1, 1).reshape(nfmodel, 1), np.zeros((nfmodel, 1)), axis=1)
    tempdispCF = tempdispCF.T

    # Matrix reshape
    H = H.reshape(1, np.size(H))
    f = f.reshape(1, np.size(f))
    ts = ts.reshape(np.size(ts), 1)
    H = col2row(H, 1)
    f = col2row(f, 1)
    ts = col2row(ts, 2)
    for i in range(nfmodel):
        tempdispCF[1, i] = V[i]
    indCFmode = np.arange(1, 51, 1)
    index = np.where(f < 0.5 / dt)
    index = index[0]
    indey = np.where(
        (f[index, 0] >= tempdispCF[0, indCFmode[0] - 1]) & (f[index, 0] <= tempdispCF[0, indCFmode[49] - 1]))
    indCwm = indey[0]
    nfm = np.size(indCwm)
    indf1 = indCwm[0]
    indf2 = indCwm[nfm - 1]
    f2 = interpolate.interp1d(np.linspace(f[indf1, 0], f[indf2, 0], 50), V, 'linear')
    Cw = f2(f[indCwm, 0]).reshape(nfm, 1)
    Kw = 2 * cmath.pi * f[indCwm, 0] / Cw[:, 0]
    Kw = Kw.reshape(nfm, 1)
    dist = np.zeros((1, ns))
    ps1 = np.zeros((nfm, ns), dtype=complex)
    ps23 = np.zeros((nfm, ns), dtype=complex)
    tempSR = np.zeros((nfm, ns), dtype=complex)
    c = np.zeros((nfm, ns), dtype=complex)
    d = np.zeros((nfm, ns), dtype=complex)
    e = np.zeros((nfm, ns), dtype=complex)
    for i in range(nfm):
        c[i, :] = ts[:, :]
    for j in range(ns):
        d[:, j] = f[indCwm, 0]
    ps1 = -1j * 2 * cmath.pi * d * c
    aq = 1.0 / 100
    U = U.T

    for k in range(nr):
        dist[0, :] = D[:, k].T
        for i in range(nfm):
            c[i, :] = dist[:, :]
        for j in range(ns):
            d[:, j] = H[indCwm, 0]
            e[:, j] = Kw[:, 0]
        tempSR = d * (1 / (c ** 0.5))
        ps23 = (-1j - aq / 2) * e * c
        U[k, indCwm] = U[k, indCwm] + (tempSR * np.exp(ps1 + ps23)).sum(axis=1).reshape(1, nfm)

    U = U.T
    # Apply frequency-domain symmetries
    for i in range(int(math.floor(nf + 1) / 2) - 1):
        U[nf - 1 - i, :] = np.conj(U[i + 1, :].T)
    uxt = ifft(U.T).T
    uxt = uxt.real
    # St = uxt  # Without normalization
    St = uxt / (uxt.max(axis=0))  # Normalize data
    # Add white noise
    sigma = 0.05
    St = St + sigma * np.random.randn(St.shape[0], St.shape[1])
    St = St / (St.max(axis=0))  # Normalize data
    #np.save(file=os.path.join('data_pre/train data', 'image(surface wave)%d.npy' % (aa+1)), arr=St)
    aa = aa + 1

    ## Visualization
    plt.figure(figsize=(10, 6))
    example_data = St[:, 0]
    time_axis = np.arange(0, nn) * dt
    plt.plot(time_axis, example_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('A pair-wheel seismic signal')
    plt.legend()
    plt.show()
