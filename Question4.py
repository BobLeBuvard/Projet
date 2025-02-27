from SimTABS import kelvin,calculTemperaturesEuler
import RechercheRacine 

#______________________________________________________________________________________________________#
# question 4.1 base



def T_optimale(T_room, T_surface):
    """Calcule la température de confort selon la norme"""
    return (T_room + T_surface) / 2

def EstTemperatureOK(temps, T_room, T_surface):
    """Vérifie si la température est dans la plage de confort pendant les heures de bureau"""
    HeuresBureau = [8, 19]
    EN15251_temp = [19.5, 24]  # En °C, pas besoin de kelvin()
    
    if temps < HeuresBureau[0] or temps > HeuresBureau[1]:
        return False  # Hors des heures de bureau
    
    Temp_optimale = T_optimale(T_room, T_surface)
    return EN15251_temp[0] <= Temp_optimale <= EN15251_temp[1]

def T_confort_max(FenetreDeTemps, T0, h):
    """Calcule la température de confort maximale en testant différents scénarios de chauffage"""
    delta_t = 0  # Initialisation du temps de chauffage
    
    while delta_t < 24:
        delta_t += 0.5  # On teste par pas de 30 min
        
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h)  # Simulation de température
        
        T_confort_values = []  # Stocke les valeurs de T_confort sur 24h
        
        for i in range(len(t)):  # On parcourt chaque instant simulé
            if EstTemperatureOK(t[i], T[i, 0], T[i, 4]):  
                T_confort_values.append(T_optimale(T[i, 0], T[i, 4]))
        
        if T_confort_values:
            Tmax_confort = max(T_confort_values)  # Trouver le maximum
            print(f"Delta_t = {delta_t}h -> Tmax_confort = {Tmax_confort:.2f}°C")
    
    return Tmax_confort  # Retourne la valeur maximale atteinte

