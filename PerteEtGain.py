import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
PERTE_ET_GAIN = np.loadtxt('PerteEtGain.txt') # array constante du fichier texte 
heures = PERTE_ET_GAIN[0]
flux = PERTE_ET_GAIN[1]
val = scp.interpolate.CubicSpline(heures,flux,bc_type='clamped')
val2 = scp.interpolate.UnivariateSpline(heures,flux,s = 5)
plt.plot(heures, val(heures),label='Spline')
plt.plot(heures, val2(heures),label='Spline')
plt.show()