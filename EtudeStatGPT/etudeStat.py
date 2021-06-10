import pickle

def main():
    #main variables
    CLEAN_FILE = "./Data/CleanData.txt"
    data = pickle.load(open(CLEAN_FILE, 'rb'))
    LENGTH = range(3, 18)    #sentence length
    NUMBER = 200            #number of phrases per dict
    inputLength= range(2,17)    #length of input, indepent from sentence length

    #stat variables
    length_repartition = {i: 0 for i in range(19)}
    true_length_repartition = {i: 0 for i in range(19)}
    number_preds = 0
    true_repartition_by_length = {i: [] for i in range(19)}     #tous les scores en pourcentages
    all_scores = []                                             #tous les scores en fonction de la longueur
    all_preds = []
    number_true_preds = 0
    per_preds_in_data = 0

    for inputLen in inputLength:
        for leng in LENGTH:
            if inputLen >= leng:
                continue
            OUTPUT_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
            score = pickle.load(open(OUTPUT_FILE, 'rb'))

            length_repartition = predLength(score, length_repartition)
            number_preds = totalPredicts(score, number_preds)
            true_length_repartition = predTrueLength(score, true_length_repartition)
            true_repartition_by_length = predTrueRepartitionByLength(score, true_repartition_by_length)
            all_scores = allScores(score, all_scores)
            all_preds = numberDiffPred(score, all_preds)
            number_true_preds = numberTruePreds(score, number_true_preds)
    #per_preds_in_data = percPredsOfBDD(data, number_true_preds)
    print("nombre total de prédictions :", number_preds)
    print("nombre total de prédictions correctes:", number_true_preds)
    print("nombre prédictions différentes:", len(all_preds))

    print("repartition des longueurs des prédictions :",length_repartition)
    print("repartition des longueurs des prédictions correctes :", true_length_repartition)

    print("pourcentage de phrases prédites dans la BDD :",per_preds_in_data)

    return

def predLength(scores, length_repartition): #répartition du nombre de prédictions en fonction de la longueur
    for score in scores:
        length_repartition[score[1]]+=1
    return length_repartition

def predTrueLength(scores, true_length_repartition):    #répartition du nombre de prédictions justes en fonction de la longueur
    for score in scores:
        if score[1] == score[2]:
            true_length_repartition[score[1]]+=1
    return true_length_repartition

def totalPredicts(score, number_preds):   #nombre total de prédictions
    number_preds+=len(score)
    return number_preds

def predTrueRepartitionByLength(scores, true_repartition_by_length):    #répartition du score de la prédiction, séparées par leur longueur totale en nombre de mots justes
    for score in scores:
        true_repartition_by_length[score[1]].append(score[2])
    return true_repartition_by_length

def allScores(scores, all_scores):      #score de chaque prédiction en pourcentage en fonction de la longueur des prédictions
    for score in scores:
        perc = score[2]/score[1]
        all_scores.append(perc)
    return all_scores

def numberDiffPred(scores, dict):       #dictionnaire des prédictions, toutes sont différentes
    for score in scores:
        if score[0] not in dict:
            dict.append(score[0])
    return dict

def numberTruePreds(scores, number_true_preds):    #nombre de prédictions justes
    for score in scores:
        if score[1] == score[2]:
            number_true_preds+=1
    return number_true_preds

def percPredsOfBDD(clean_data, number_true_preds):  #proportion de phrases de la BDD que cela représente
    number_total = 0
    for dict in clean_data:
        number_total+=len(dict)
    return  number_true_preds/number_total


#-------------------------------------------------------------------------
def fileStat(inputLen, expectedLen, n_samples=200):
    print()
    print("expectedLen", expectedLen)
    #stats variables
    length_repartition = {i: 0 for i in range(19)}
    true_length_repartition = {i: 0 for i in range(19)}
    number_preds = 0
    true_repartition_by_length = {i: [] for i in range(19)}  # tous les scores en pourcentages
    all_scores = []  # tous les scores en fonction de la longueur
    all_preds = []
    number_true_preds = 0
    per_preds_in_data = 0

    OUTPUT_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(expectedLen) + "_" + str(n_samples) + "_" + str(inputLen) + ".txt"
    score = pickle.load(open(OUTPUT_FILE, 'rb'))

    length_repartition = predLength(score, length_repartition)
    number_preds = totalPredicts(score, number_preds)
    true_length_repartition = predTrueLength(score, true_length_repartition)
    true_repartition_by_length = predTrueRepartitionByLength(score, true_repartition_by_length)
    all_scores = allScores(score, all_scores)
    all_preds = numberDiffPred(score, all_preds)
    number_true_preds = numberTruePreds(score, number_true_preds)

    print("nombre total de prédictions :", number_preds)
    print("nombre total de prédictions correctes:", number_true_preds)
    print("nombre prédictions différentes:", len(all_preds))

    print("repartition des longueurs des prédictions :", length_repartition)
    print("repartition des longueurs des prédictions correctes :", true_length_repartition)


    return

#-------------------------------------------------------------------------
for length in range(9, 18):
    fileStat(8, length)