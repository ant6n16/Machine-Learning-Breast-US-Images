def matriz_confusion(prediccion, GT_test):
    
    import numpy as np
    # Generamos la matriz de confusion de 2x2
    Mt_confusion = np.zeros((2,2)); VP = 0; VN = 0; FP = 0; FN = 0;

    # Matriz auxiliar para averiguar VP, VN, FP, FN
    aux = np.zeros((len(prediccion),2)); aux[prediccion==0] = [1., 0.]; aux[prediccion==1] = [0., 1.]

    # Comparamos la clasificacion con la verdad de referencia de test
    comp = (aux == GT_test)[:,0]

    for k in range(len(comp)):

        # Si es igual a la verdad de referencia tendremos los verdaderos negativos en caso de que el 1 esté en la
        # primera columna y los verdaderos positivos en el otro caso (el 1 estaría en la segunda columna)
        if comp[k]:
            if (GT_test[k] == [1., 0.])[0]:
                VN = VN+1
            else:
                VP = VP+1  

        # Si es distinto a la verdad de referencia tendremos los falsos negativos en caso de que el 1 esté en la
        # primera columna y los falsos positivos en el otro caso (el 1 estaría en la segunda columna)        
        else:
            if (GT_test[k] == [1., 0.])[0]:
                FP = FP+1
            else:
                FN = FN+1

    # Incorporamos VP, VN, FP, FN a la matriz de confusion
    Mt_confusion[0,0] = VP
    Mt_confusion[0,1] = FP
    Mt_confusion[1,0] = FN
    Mt_confusion[1,1] = VN    
    
    return Mt_confusion



def metricas(Mt_confusion):
    
    VP = Mt_confusion[0,0]
    FP = Mt_confusion[0,1]
    FN = Mt_confusion[1,0]
    VN = Mt_confusion[1,1]
    
    Acuraccy = (VP+VN)/(VP+FP+FN+VN)
    Sensibilidad = VP/(VP+FN)
    Especificidad = VN/(VN+FP)
    VPP = VP/(VP+FP) # Valor predictivo positivo
    VPN = VN/(VN + FN) # Valor predictivo negativo
    F1 = 2*Sensibilidad*VPP / (Sensibilidad+VPP)

    print("Matriz de confusion: "+str(Mt_confusion)+"\n")
    print("Accuracy: "+str(Acuraccy))
    print("Sensibilidad: "+str(Sensibilidad))
    print("Especificidad: "+str(Especificidad))
    print("F1 Score: "+str(F1))
    print("Valor Predictivo Positivo: "+str(VPP))
    print("Valor Predictivo Negativo: "+str(VPN))