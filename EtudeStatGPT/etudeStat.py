import pickle

def main(dif=False):
    if dif == True:
        jump=1
    else:
        jump=0

    #main variables
    CLEAN_FILE = "./Data/CleanData.txt"
    OUTPUT_FILE = "./Predictions/globalScoreResult.txt"
    data = pickle.load(open(CLEAN_FILE, 'rb'))
    LENGTH = range(3, 18)    #sentence length
    NUMBER = 200            #number of phrases per dict
    inputLength= range(2,17)    #length of input, indepent from sentence length

    #stat variables
    length_repartition = {i: 0 for i in range(20)}
    true_length_repartition = {i: 0 for i in range(20)}
    number_preds = 0
    true_repartition_by_length = {i: [] for i in range(20)}     #tous les scores en pourcentages
    all_scores = []                                             #tous les scores en fonction de la longueur
    all_preds = []
    number_true_preds = 0
    per_preds_in_data = 0

    for inputLen in inputLength:
        for leng in LENGTH:
            print()
            print("phrase length :", leng)
            print("input length:", inputLen)
            if inputLen >= leng-jump:
                continue
            SCORE_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
            score = pickle.load(open(SCORE_FILE, 'rb'))

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

    #print("pourcentage de phrases prédites dans la BDD :",per_preds_in_data)

    scores = (number_preds, number_true_preds, len(all_preds), length_repartition, true_length_repartition)

    with open(OUTPUT_FILE, 'wb') as fp:
        pickle.dump(scores, fp)

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
    length_repartition = {i: 0 for i in range(20)}
    true_length_repartition = {i: 0 for i in range(20)}
    number_preds = 0
    true_repartition_by_length = {i: [] for i in range(20)}  # tous les scores en pourcentages
    all_scores = []  # tous les scores en fonction de la longueur
    all_preds = []
    number_true_preds = 0
    per_preds_in_data = 0

    SCORE_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(expectedLen) + "_" + str(n_samples) + "_" + str(inputLen) + ".txt"
    score = pickle.load(open(SCORE_FILE, 'rb'))
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


    return number_preds

def InputStat(input):
    nb = 0
    for length in range(input + 1, 18):
        nb+=fileStat(input, length)
    print("nb de pred pour input=", input, " :", nb)
#-------------------------------------------------------------------------

#main(dif=False)

InputStat(14)