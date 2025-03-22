import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
PERTE_ET_GAIN = np.loadtxt('PerteEtGain.txt') # array constante du fichier texte 
heures = PERTE_ET_GAIN[0]
flux = PERTE_ET_GAIN[1]
val = scp.interpolate.CubicSpline(heures,flux,bc_type='periodic')

if __name__ == "__main__":  # nécessaire pour tester le graphe si on débug
    plt.plot(heures, val(heures),label='données interpolées')
    plt.plot(PERTE_ET_GAIN[0],PERTE_ET_GAIN[1],'.', label = 'données brutes')
    plt.xlabel("Heures")  # Nom de l'axe des X
    plt.ylabel("Flux")  # Nom de l'axe des Y
    plt.title("Variation du flux en fonction du temps")  # Titre du graphe
    plt.legend(loc = 'best', fontsize = 8)
    plt.show()     # Louis a besoin de plt.show() pour afficher le graph
def g(t):    
    return val(t)