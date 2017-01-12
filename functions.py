import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import MinMaxScaler
from operator import eq

random.seed(34898)

def mean_value(folders):
    val = []
    for folder in folders:
        for f in folder:
            n = pd.read_csv(f, header=None, dtype=np.float32).values.shape[0]
            if n > 90:
                val.append(n)
    return sum(val)/len(val) + 10


def interpolate(data, N):
    data_len = len(data)
    diff = abs(N - data_len)
    if N > data_len:
        step = max([math.floor(data_len/diff), 1])  # max serve ad evitare che il passo sia 0
        for i in range(1, diff + 1):
            x = i * step - 1
            new_value = np.add(data[int(x-1)], data[int(x)])
            new_value = np.divide(new_value, 2)
            data = np.insert(data, int(x), new_value, 0)
    elif N < data_len:  # se il valore da interpolare e' minore della lunghezza dei dati, tolgo delle righe all'inizio e alla fine
        m = diff / 2
        if diff % 2 == 0:  # se diff e' pari
            data = data[m:data_len - m, :]
        else:  # se diff e' dispari
            data = data[m:data_len - m - 1, :]
    return data



def import_data(folders, N):
    # tutti i dati verranno interpolati con la lunghezza della media delle lunghezze
    dataset = []
    labels = []
    label = 0
    for folder in folders:
        #random.shuffle(folder)  # disordino i dati
        count = 0  # contatore che indica se il file va in train o in test
        for f in folder:
            temp = pd.read_csv(f, header=None, dtype=np.float32)
            temp = temp.values

            # NON SONO SICURO CHE SI DEBBA FARE
            scaler = MinMaxScaler(feature_range=(0, 1))
            temp = scaler.fit_transform(temp)
            ##################################

            temp = interpolate(temp, N)

            dataset.append(temp)
            labels.append(label)
            count += 1  # aumento il contatore
        label += 1

    # disordino le due liste (nella stessa maniera)
    c = list(zip(dataset, labels))
    random.shuffle(c)
    dataset, labels = zip(*c)

    limit = int(len(dataset) * 0.8)  # il 80% dei valori andranno il train, il 20% in test
    datasetTrain = dataset[:limit]
    labelsTrain = labels[:limit]
    datasetTest = dataset[limit:]
    labelsTest = labels[limit:]

    return np.array(datasetTrain), np.array(datasetTest), np.array(labelsTrain), np.array(labelsTest)

def stat(pred, corretto, Nlabel):
    lab = ['thankyou', 'maybe', 'never', 'wonderful', 'baby', 'world', 'rainbow', 'little']
    pred = pred.tolist()
    corretto = corretto.tolist()

    N = len(pred)
    c = map(eq, pred, corretto)
    true_value = c.count(True)
    false_value = N - true_value
    count = [0 for i in range(0,Nlabel)]
    count_real = [0 for i in range(0,Nlabel)]

    for i in range(0,Nlabel):
        count_real[i] = corretto.count(i)
    print "Valori totali: " + str(N)
    print "Valori indovinati: " + str(true_value)
    print "Valori errati: " + str(false_value)
    print "Percentuale indovinati: " + str(100.0 * true_value/N) + "%"

    for i in range(0,N):
        if corretto[i] == pred[i]:
            j = corretto[i]
            count[j] = count[j] + 1

    for i in range(0,Nlabel):
        print "Il movimento " + lab[i] + " e' stato indovinato " + str(count[i]) + "/" + str(count_real[i]) + " volte.  " + str(100.0 * count[i] / count_real[i]) + "%"



