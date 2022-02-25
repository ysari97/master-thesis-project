import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('lsv_rel_KafueGorgeLower.txt')
dataka=np.loadtxt('lsv_rel_Kariba.txt')

level=data[0,:]
storage=data[2,:]

level_ka=dataka[0,:]
storage_ka=dataka[2,:]

#plt.plot(level,storage)
plt.plot(level_ka,storage_ka)
plt.show()
