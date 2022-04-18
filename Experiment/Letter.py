import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from scipy import stats
sys.path.append('..')
from core_utils import OptimumHyperparamFinder

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')


def main(generation_type):
    print('=========Testing Letter dataset=========')
    df = pd.read_table('letter-recognition.data', sep=',', header=None)
    X_data = df.drop(0, axis=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_data)
    scaled = pd.DataFrame(scaler.transform(X_data))
    scaled['class'] = df[0]
    df = scaled
    classes = df['class'].unique()

    normal = ['T', 'I', 'D', 'N', 'G', 'S', 'B', 'A', 'J', 'M', 'X', 'O', 'R']
    out = ['F', 'C', 'H', 'W', 'L', 'P', 'E', 'V', 'Y', 'Q', 'U', 'K', 'Z']
    indice = []
    outIndice = []
    for letter in normal:
        indice += df['class'].index[df['class'] == letter].tolist()

    for letter in out:
        outIndice += df['class'].index[df['class'] == letter].tolist()
    indice = np.sort(indice)
    outIndice = np.sort(outIndice)

    scores_Letter_macro = []
    scores_Letter_micro = []
    scores_Letter_weighted = []
    MCC = []

    for iter_ in range(10):
        print('iter : ', iter_)

        target1 = df.iloc[indice]
        target_train1 = target1.sample(frac=0.7)
        target_val1 = target1.drop(target_train1.index)
        target_train1 = target_train1.reset_index(drop=True)
        target_val1 = target_val1.reset_index(drop=True)
        outliers1 = df.iloc[outIndice].reset_index(drop=True)
        testset1 = pd.concat([target_val1, outliers1]).reset_index(drop=True)

        y_true1 = [1 if i in normal else -1 for i in testset1['class']]
        target_train1 = target_train1.drop('class', axis=1)
        testset1 = testset1.drop('class', axis=1)

        model1 = OptimumHyperparamFinder(target_train1,generation_type)
        opt_comb1 = model1.search_optimal_hyperparameters()

        clf1 = OneClassSVM(nu=opt_comb1[0], gamma=opt_comb1[1]).fit(target_train1)
        y_pred1 = clf1.predict(testset1)
        score1_macro = f1_score(y_true1, y_pred1, average='macro')
        score1_micro = f1_score(y_true1, y_pred1, average='micro')
        score1_weighted = f1_score(y_true1, y_pred1, average='weighted')
        mcc1 = matthews_corrcoef(y_true1, y_pred1)

        target2 = df.iloc[outIndice]
        target_train2 = target2.sample(frac=0.7)
        target_val2 = target2.drop(target_train2.index)
        target_train2 = target_train2.reset_index(drop=True)
        target_val2 = target_val2.reset_index(drop=True)
        outliers2 = df.iloc[indice].reset_index(drop=True)
        testset2 = pd.concat([target_val2, outliers2]).reset_index(drop=True)

        y_true2 = [1 if i in out else -1 for i in testset2['class']]
        target_train2 = target_train2.drop('class', axis=1)
        testset2 = testset2.drop('class', axis=1)

        model2 = OptimumHyperparamFinder(target_train2,generation_type)
        opt_comb2 = model2.search_optimal_hyperparameters()

        clf2 = OneClassSVM(nu=opt_comb2[0], gamma=opt_comb2[1]).fit(target_train2)
        y_pred2 = clf2.predict(testset2)
        score2_macro = f1_score(y_true2, y_pred2, average='macro')
        score2_micro = f1_score(y_true2, y_pred2, average='micro')
        score2_weighted = f1_score(y_true2, y_pred2, average='weighted')
        mcc2 = matthews_corrcoef(y_true2, y_pred2)

        mean_score_macro = np.mean([score1_macro, score2_macro])
        mean_score_micro = np.mean([score1_micro, score2_micro])
        mean_score_weighted = np.mean([score1_weighted, score2_weighted])
        scores_Letter_macro.append(mean_score_macro)
        scores_Letter_micro.append(mean_score_micro)
        scores_Letter_weighted.append(mean_score_weighted)
        MCC.append(np.mean([mcc1, mcc2]))

    _, pval_f1 = stats.ttest_1samp(scores_Letter_macro, 0.601)
    _, pval_MCC = stats.ttest_1samp(MCC, 0.349)
    with open('TestResult.txt', 'a') as f:
        f.write('Letter : ' + str(np.mean(scores_Letter_macro)) + ', p_f1 : ' + str(pval_f1) + \
                ', MCC : ' + str(np.mean(MCC)) + ', p_MCC : ' + str(pval_MCC) + '\n')
    print('Letter : ', np.mean(scores_Letter_macro))

