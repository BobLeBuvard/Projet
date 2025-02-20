import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
degre_de_lissage = 5
    
if __name__ == "__main__":  # nécessaire pour tester le graphe si on débug
    
    PERTE_ET_GAIN = np.loadtxt('PerteEtGain.txt') # array constante du fichier texte 
    heures = PERTE_ET_GAIN[0]
    flux = PERTE_ET_GAIN[1]
    val = scp.interpolate.CubicSpline(heures,flux,bc_type='clamped')
    val2 = scp.interpolate.UnivariateSpline(heures,flux,s = 10)
    #plt.plot(heures, val(heures),label='Spline')
    plt.plot(heures, val2(heures),label='Spline')
    plt.show()
def g(x):    
    PERTE_ET_GAIN = np.loadtxt('PerteEtGain.txt') # array constante du fichier texte 
    
    heures = PERTE_ET_GAIN[0]
    flux = PERTE_ET_GAIN[1]
    val = scp.interpolate.UnivariateSpline(heures,flux,s = degre_de_lissage)
    return val(x)