import math
import cmath
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import interpolate
import matplotlib.pyplot as plt
import random
import pylab
from SeismicCurve import SeismicCurve
from col2row import col2row
import os

# Basic parameter settings
Fs = 50
dt = 0.001
A = 10 ** 3  # Amplitude

# Source and receiver positions
ns = 2  # Number of sources (e.g., wheels)
nr = 1  # Number of receivers
rx = 0  # Receiver x-coordinate
ry = 0  # Receiver y-coordinate

sx = 0
sy = [5, 7]  # Source y-coordinates (two wheels)
D = np.zeros([ns, nr])  # Source-receiver distance
ts = np.zeros([1, ns])  # Source time delay

# Calculate source-receiver distances
for i in range(nr):
    for k in range(ns):
        D[k, i] = ((rx - sx) ** 2 + (ry - sy[k]) ** 2) ** 0.5

# Ricker wavelet generator
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

# Load fundamental Rayleigh wave velocity
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

for i in range(5):
    fm = 5 + 10 * random.random()
    V_car = (35 + 15 * random.random())/3.6  # Vehicle speed (m/s)
    ts[0, 0] = 4.5 * random.random()
    ts[0, 1] = ts[0, 0]  # Both axles share the same delay

    # Single axle wavelet
    wavelet = Ricker(fm, dt)

    waveletnt = np.size(wavelet)

    # Combine two axles
    D_train = np.array([0, 3])  # Axle positions (meters apart)

    Dnt = np.zeros([1, 2])
    for i in range(2):
        Dnt[0, i] = math.floor(D_train[i] / V_car / dt)
    wavelet_per_carriage = np.zeros([1, waveletnt + int(Dnt.max(axis=1)[0]) + 1])
    for i in range(np.size(D_train)):
        wavelet_wheel = np.zeros([1, waveletnt + int(Dnt.max(axis=1)[0]) + 1])
        for j in range(waveletnt):
            wavelet_wheel[0, j + int(Dnt[0, i])] = wavelet[0, j]
        wavelet_per_carriage = wavelet_per_carriage + wavelet_wheel


    source = wavelet_per_carriage
    waveletnt = np.size(wavelet)
    # Pad the signal to match total simulation time
    source = np.append(source, np.zeros([1, nn - np.size(source)]))
    H = fft(source)
    nf = np.size(source)
    f = 1 / dt / nf * np.arange(0, nf, 1)

    # Initialize frequency-domain response
    U = np.zeros((nn, nr), dtype=complex)
    nfmodel = np.size(V)
    tempdispCF = np.append(np.arange(1, nfmodel + 1, 1).reshape(nfmodel, 1), np.zeros((nfmodel, 1)), axis=1)
    tempdispCF = tempdispCF.T

    # matrix reshape
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

    St = uxt  # Final signal without normalization
    # St = uxt / (uxt.max(axis=0))  # Optionally normalize

    # Add white Gaussian noise
    sigma = 0.05
    St = St + sigma * A * np.random.randn(St.shape[0], St.shape[1])

    # np.save(file=os.path.join('data_pre/train data', 'image(car)%d.npy' % (aa + 1)),
    #      arr=St)
    # aa = aa + 1
    ##visualization
    plt.figure(figsize=(10, 6))
    example_data = St[:, 0]
    time_axis = np.arange(0, nn) * dt
    plt.plot(time_axis, example_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('A car signal')
    plt.legend()
    plt.show()
    plt.close('all')
