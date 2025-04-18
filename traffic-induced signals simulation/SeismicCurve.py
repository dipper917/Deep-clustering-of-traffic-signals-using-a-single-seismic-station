import numpy as np
import matplotlib.pyplot as plt

def SeismicCurve(data, dt, line=1.0, off=1, scale=1):
    # Normalize data for consistent amplitude visualization
    data = data / (scale * data.max(axis=0))
    x = np.arange(data.shape[0]) * dt
    # Add a zero trace to the start for alignment
    data = np.append(np.zeros((data.shape[0], 1)), data, axis=1)
    # Plot each trace as a waveform
    for i in range(data.shape[1]):
        y = data[:, i] + i * off
        y = y.reshape(x.shape)
        plt.plot(y, x, 'k', linewidth=line)

    # Configure plot axes
    ax = plt.gca()
    plt.ylim(0, data.shape[0] * dt)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    plt.xlim(0, data.shape[1])
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
    plt.xlabel('Trace number', font)
    plt.ylabel('Time(s)', font)
    plt.yticks(rotation=90)
    plt.tick_params(labelsize=8)
    plt.tight_layout()

    return
