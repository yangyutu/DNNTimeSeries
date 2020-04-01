from Data.SimpleSineWave.SineWaveData import SineWaveData
import matplotlib.pyplot as plt


data = SineWaveData(1000, 200, 0, seed = 1)

x, y = data.fetch_batch(3)


plt.figure(1)

for e in x:
    plt.plot(e)

plt.show()


data = SineWaveData(1000, 200, 0, seed = 1, noiseStrength= 0.1, noiseFlag=True)

x, y = data.fetch_batch(3)


plt.figure(2)

for e in x:
    plt.plot(e)

plt.show()