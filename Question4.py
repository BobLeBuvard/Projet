from SimTABS import kelvin,calculTemperaturesEuler
import RechercheRacine 

#______________________________________________________________________________________________________#
# question 4.1 base



def T_optimale(T_room, T_surface):
    '''calcule la température ressentie en fonction de la chaleur de la pièce et celles des surfaces'''
    return((T_room+T_surface)/2)
def EstTemperatureOK(temps,T_room,T_surface):
    HeuresBureau = [8,19]
    EN15251_temp = kelvin([19.5,24])
    if (temps < HeuresBureau[0] or temps >HeuresBureau[1]):
        return False
    Temp_optimale = T_optimale(T_room,T_surface)
    if (EN15251_temp[0] <= Temp_optimale <= EN15251_temp[1]):
        return True
    else:
        return False
    

def T_confort_max(FenetreDeTemps, T0, h):
    while(delta_t <24):
        delta_t += 0.5
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h )
        for i in range(t): #tester pour tous les éléments de T
            if not EstTemperatureOK(i,T[0],T[4]): 
                break