import numpy as np
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from scipy.spatial import distance
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
import random
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import time

np.seterr(divide='ignore', invalid='ignore')

class PseudoSamples:
    def __init__(self, data, generation_type):
        # data type should be DataFrame
        # testnum : number of acceptance rejection test by uniform number
        self.nSample = data.shape[0]
        self.dim = data.shape[1]
        self.data = copy.deepcopy(data)
        N = int(5 * np.log(len(data)))
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(data)
        dists = nbrs.kneighbors(data)[0]
        self.epsilon = np.mean(np.max(dists, axis=1))
        self.testNum = int(2*np.sqrt(self.dim))
        #'sampling' or 'duplicate's
        self.generation_type = generation_type



    # only in 2D
    def visualize_data(self):
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], s=20, color='skyblue')
        '''
        for i in range(self.data.shape[0]):
            plt.text(self.data.iloc[i, 0], self.data.iloc[i, 1], str(i))
        '''
        plt.show()

    def get_neighbors(self, index_of_target):

        point_tree = cKDTree(self.data)
        arr = np.copy(self.data.iloc[index_of_target, :], order='C')
        idx = point_tree.query_ball_point(arr, self.epsilon)
        # idx = point_tree.query_ball_point(self.data.iloc[index_of_target, :], self.epsilon)
        pointset = self.data.loc[idx, :]
        target = [True if i == index_of_target else False for i in idx]
        pointset['target'] = target

        return idx, pointset

    # only in 2D
    def visualize_ball(self, index_of_target):

        idx, pointset = self.get_neighbors(index_of_target)
        color = ['red' if i == True else 'navy' for i in pointset['target']]
        targetcoord = np.squeeze(pointset[pointset['target'] == True].values)[0:self.dim]
        fig, ax = plt.subplots()

        ax.scatter(pointset.iloc[:, 0], pointset.iloc[:, 1], color=color, label='_nolegend_')
        ax.scatter(targetcoord[0], targetcoord[1], color='red', label='target')
        ax.set_xlim(targetcoord[0] - self.epsilon, targetcoord[0] + self.epsilon)
        ax.set_ylim(targetcoord[1] - self.epsilon, targetcoord[1] + self.epsilon)
        # plt.plot(df['x1'], df['x2'], 'ro')
        for i in range(pointset.shape[0]):
            ax.text(pointset.iloc[i, 0], pointset.iloc[i, 1], str(idx[i]), fontsize=10)

        ax.scatter(np.mean(pointset.iloc[:, 0]), np.mean(pointset.iloc[:, 1]), s=30, color='yellow',
                   label='centroid of neighbors')
        ax.text(np.mean(pointset.iloc[:, 0]), np.mean(pointset.iloc[:, 1]), str('centroid of neighbors'), fontsize=15)
        # ax.text(targetcoord[0], targetcoord[1]  , str('target_sample'), fontsize=15)
        ax.legend(loc=2)
        circle = plt.Circle((targetcoord[[0]], targetcoord[[1]]), radius=self.epsilon, color='g', fill=False)
        ax.add_artist(circle)
        fig.show()

    def get_features(self):

        # f1 : epsilon 내에 잡힌 neighbor target의 갯수; volume 추정량
        # f2 : neighbor target들의 중심과 target point 의 거리
        # f3 : outlier 확정 여부
        f1 = []
        f2 = []
        f3 = []
        target_candidates = []
        outlier_candidates = []
        for i in range(len(self.data)):
            idx, pointset = self.get_neighbors(i)
            targetcoord = np.squeeze(pointset[pointset['target'] == True].values)[0:self.dim]

            f1_ = len(idx)
            f2_ = np.nan if len(idx) == 1 else distance.euclidean(targetcoord, np.mean(pointset.iloc[:, 0:self.dim]))
            f3_ = 'outlier' if len(idx) == 1 else 'normal'
            if len(idx) != 1:
                #generation directions with sampling with replace
                pseudo_outlier = []
                for _ in range(2*self.testNum):
                    if self.generation_type == 'replace':
                        p_o = targetcoord - np.mean(pointset.sample(frac=1, replace = True).iloc[:, 0:self.dim])
                    elif self.generation_type == 'sampling':
                        p_o = targetcoord - np.mean(pointset.sample(frac=0.8).iloc[:, 0:self.dim])
                    p_o = self.epsilon * (p_o / np.linalg.norm(p_o)) + targetcoord
                    pseudo_outlier.append(p_o)
            else:
                pseudo_outlier = [targetcoord]

            '''
            if len(idx) != 1:
                pseudo_target = []
                for _ in range(2*self.testNum):
                    try:
                        #density gradient만 살짝 바꾸어 가고 neighbor는 그대로 둔다
                        p_t = np.array(np.mean(pointset.sample(frac=1, replace=True).iloc[:, 0:self.dim]) - targetcoord, dtype=float)
                        p_t = p_t / np.linalg.norm(p_t)
                        neighbors = np.array(pointset.iloc[:, 0:self.dim] - targetcoord, dtype=float)
                        projections = np.matmul(neighbors, p_t)
                        p_t = p_t * min(projections[projections > 0]) + targetcoord
                    except ValueError:
                        p_t = [np.nan] * self.dim
                    pseudo_target.append(p_t)
            else:
                pseudo_target = [[np.nan] * self.dim]
            '''
            if len(idx) != 1:
                try:
                    pseudo_target = np.array(np.mean(pointset.iloc[:, 0:self.dim]) - targetcoord, dtype=float)
                    pseudo_target = pseudo_target / np.linalg.norm(pseudo_target)
                    neighbors = np.array(pointset.iloc[:, 0:self.dim] - targetcoord, dtype=float)
                    projections = np.matmul(neighbors, pseudo_target)

                    pseudo_target = pseudo_target * min(projections[projections > 0]) + targetcoord
                except ValueError:
                    pseudo_target = [np.nan] * self.dim
            else:
                pseudo_target = [np.nan] * self.dim


            f1.append(f1_)
            f2.append(f2_)
            f3.append(f3_)
            outlier_candidates.append(pseudo_outlier)
            target_candidates.append(pseudo_target)

        f1 = (np.array(f1) - min(f1)) / (max(f1) - min(f1))  # epsilon을 너무 크게잡으면 값이 모두 같아져서 min(f1)값이 없어 에러발생
        if sum(np.isnan(f1)) == len(f1):
            print('Too large epsilon to calculate feature 1')
        f2 = np.array(f2)
        f2 = (f2 - min(f2[np.isnan(f2) == False])) / (max(f2[np.isnan(f2) == False]) - min(f2[np.isnan(f2) == False]))

        prob = np.array([np.nan if f3[i] == 'outlier' else np.float32(f2[i] - f1[i]) for i in range(len(f1))])
        max_ = max(prob[np.isnan(prob) == False])
        min_ = min(prob[np.isnan(prob) == False])
        prob = np.array([1 if np.isnan(prob[i]) else (prob[i] - min_) / (max_ - min_) for i in range(len(prob))])

        return f1, f2, f3, outlier_candidates, target_candidates

    def get_pseudo_points(self):

        f1, f2, f3, outlier_candidates, target_candidates = self.get_features()

        prob = np.array([np.nan if f3[i] == 'outlier' else (np.float32(f2[i] - f1[i])) for i in range(len(f1))])
        max_ = max(prob[np.isnan(prob) == False])
        min_ = min(prob[np.isnan(prob) == False])
        prob = np.array([1 if np.isnan(prob[i]) else (prob[i] - min_) / (max_ - min_) for i in range(len(prob))])


        prob = prob.reshape(len(prob), 1)
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(prob)
        except:
            pass
        prob = np.squeeze(scaler.transform(prob))
        less_idx = prob < 0.5
        over_idx = prob >= 0.5
        less = prob[prob < 0.5]
        over = prob[prob >= 0.5]
        prob[less_idx] = less**2
        prob[over_idx] = np.sqrt(over)

        self.data['prob'] = prob

        pseudo_outliers = np.empty((0, self.dim), dtype=float)
        pseudo_targets = np.empty((0, self.dim), dtype=float)

        for i in range(self.testNum):
            unif = np.random.uniform(size=len(prob))
            #사용된 outlier,target은 list에서 지운다
            for j in range(len(unif)):
                u = unif[j]
                p = prob[j]
                if p < u:

                    accepted_target = np.expand_dims(target_candidates[j], 0)
                    pseudo_targets = np.append(pseudo_targets, accepted_target, axis=0)

                    '''
                    if len(target_candidates[j]) > 1:
                        r = random.randint(1, len(target_candidates[j]))
                        accepted_target = np.expand_dims(target_candidates[j][r-1], 0)
                        target_candidates[j].pop(r-1)
                    else:
                        accepted_target = np.array(target_candidates[j]).reshape(1, self.dim)
                    pseudo_targets = np.append(pseudo_targets, accepted_target, axis=0)
                    '''
                else:
                    if len(outlier_candidates[j]) > 1:
                        r = random.randint(1, len(outlier_candidates[j]))
                        accepted_outlier = np.expand_dims(outlier_candidates[j][r-1], 0)
                        outlier_candidates[j].pop(r-1)
                    else:
                        accepted_outlier = outlier_candidates[j][0].reshape(1, self.dim)
                    pseudo_outliers = np.append(pseudo_outliers, accepted_outlier, axis=0)

            #accepted_targets = target_candidates[unif>prob]
            #pseudo_targets = np.append(pseudo_targets, accepted_targets, axis=0)

        nearest_mean = np.mean(
            [np.sort(np.sqrt(np.sum((i - np.array(self.data)) ** 2, axis=1)))[3] for i in np.array(self.data)])
        point_tree = cKDTree(self.data.iloc[:, 0:self.dim])

        pseudoOutliersToRemove = []
        for i, o in enumerate(pseudo_outliers):
            idx = point_tree.query_ball_point(o, nearest_mean)
            if len(idx) > 1:
                pseudoOutliersToRemove.append(i)
            if sum(np.isnan(list(o))) == self.dim:
                pseudoOutliersToRemove.append(i)

        pseudoTargetsToRemove = []
        for i, t in enumerate(pseudo_targets):
            if sum(np.isnan(list(t))) == self.dim:
                pseudoTargetsToRemove.append(i)

        pseudo_outliers = np.delete(pseudo_outliers, pseudoOutliersToRemove, 0)

        pseudo_targets = np.delete(pseudo_targets, pseudoTargetsToRemove, 0)

        return np.array(pseudo_outliers, dtype='float'), np.array(pseudo_targets, dtype='float'), self.data

    def showPseudoPoints(self):

        pseudo_outliers, pseudo_targets, dataWithProb = self.get_pseudo_points()

        if self.dim == 2:
            plt.scatter(dataWithProb.iloc[:, 0], dataWithProb.iloc[:, 1], c=dataWithProb['prob'], cmap='Blues',
                        label='_nolegend_')
            plt.xlabel('V1', fontsize=18)
            plt.ylabel('V2', fontsize=18)
            plt.colorbar().set_label('Probability to edge', fontsize=15)

            # plt.plot(df['x1'], df['x2'], 'ro')
            '''
            for i in range(self.data.shape[0]):
                plt.text(self.data.iloc[i, 0], self.data.iloc[i, 1], str(i))
            '''
            plt.scatter(np.array(pseudo_outliers)[:, 0], np.array(pseudo_outliers)[:, 1], color='red', marker='x',
                        label='pseudo_outliers', s=20)
            plt.scatter(np.array(pseudo_targets)[:, 0], np.array(pseudo_targets)[:, 1], color='green', marker='*',
                        label='pseudo_targets', s=20)
            plt.legend(loc=2, fontsize='large')

            '''    
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)


            ax1.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], data = self.data, c='prob', cmap = 'Blues', label = 'originaldata')
            ax1.set_xlabel('V1', fontsize=18)
            ax1.set_ylabel('V2', fontsize=18)
            #fig.colorbar().set_label('Probability to edge', fontsize=15, ax=ax1)
            ax1.set_title('Original Data with probability to Edge')
            ax1.legend(loc=2, fontsize='large')
            # plt.plot(df['x1'], df['x2'], 'ro')



            ax2.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], data=self.data, c='prob', cmap='Blues',label='originaldata')
            ax2.set_xlabel('V1', fontsize=18)
            ax2.set_ylabel('V2', fontsize=18)
            for i in range(self.data.shape[0]):
                ax2.text(self.data.iloc[i, 0], self.data.iloc[i, 1], str(i))
            ax2.scatter(np.array(pseudo_outliers)[:, 0], np.array(pseudo_outliers)[:, 1], color='red', marker='x', label = 'pseudo_outliers', s=20)
            ax2.scatter(np.array(pseudo_targets)[:, 0], np.array(pseudo_targets)[:, 1], color='green', marker='*', label = 'pseudo_targets', s=20)
            ax2.legend(loc=2,fontsize='large')
            ax2.set_title('Original Data with Pseudo Data')

            fig.show()
            '''

    def data_with_prob(self):

        pseudo_outliers, pseudo_targets, dataWithProb = self.get_pseudo_points()

        plt.scatter(dataWithProb.iloc[:, 0], dataWithProb.iloc[:, 1], c=dataWithProb['prob'], cmap='Blues',
                    label='_nolegend_')
        '''
        plt.xlabel('x1', fontsize=18)
        plt.ylabel('x2', fontsize=18)
        '''
        plt.colorbar().set_label('Probability to edge', fontsize=15)
        plt.title('Data with probability to edge', size=20)

        # plt.plot(df['x1'], df['x2'], 'ro')
        '''
        for i in range(self.data.shape[0]):
            plt.text(self.data.iloc[i, 0], self.data.iloc[i, 1], str(i))
        '''

    def search_optimal_hyperparameters(self):

        pseudo_outliers, pseudo_targets, dataWithProb = self.get_pseudo_points()

        sigma = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10 ** 1, 10 ** 2, 10 ** 3,
                 10 ** 4]  # [10**-4, 10**-3, 10**-2, 10**-1, 1, 10**1, 10**2, 10**3, 10**4]
        nu = [0.01, 0.05, 0.1, 0.15, 0.2]  # [0.01, 0.05, 0.1, 0.15, 0.2]
        combList = [(i, j) for i in nu for j in sigma]
        errs = []
        for comb in combList:
            nu_, gamma_ = comb
            clf = OneClassSVM(nu=nu_, gamma=gamma_).fit(dataWithProb.drop('prob', axis=1))
            err_o = sum(clf.predict(pseudo_outliers) == 1) / len(pseudo_outliers)
            err_t = sum(clf.predict(pseudo_targets) == -1) / len(pseudo_targets)
            err = (err_o + err_t) / 2
            errs.append(err)

        for comb, err in zip(combList, errs):
            if err == min(errs):
                opt_comb = comb

        clf = OneClassSVM(nu=opt_comb[0], gamma=opt_comb[1]).fit(dataWithProb.drop('prob', axis=1))

        if self.dim == 2:
            plt.scatter(dataWithProb.iloc[:, 0], dataWithProb.iloc[:, 1], c=dataWithProb['prob'], cmap='Blues',
                        label='originaldata', s=20)
            plt.scatter(np.array(pseudo_outliers)[:, 0], np.array(pseudo_outliers)[:, 1], color='red', marker='x',
                        label='pseudo_outliers', s=20)
            plt.scatter(np.array(pseudo_targets)[:, 0], np.array(pseudo_targets)[:, 1], color='green', marker='*',
                        label='pseudo_targets', s=20)

            plt.title('optimal hyperparameters : nu=' + str(opt_comb[0]) + ', sigma=' + str(opt_comb[1]), size=20)

            h = .02  # step size in the mesh
            # create a mesh to plot in
            x_min, x_max = pseudo_outliers[:, 0].min() - 0.1, pseudo_outliers[:, 0].max() + 0.1
            y_min, y_max = pseudo_outliers[:, 1].min() - 0.1, pseudo_outliers[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, colors='gold', linewidths=5)  # cmap=plt.cm.Paired, color='purple')
            plt.legend(loc=2, fontsize='large')
        return opt_comb


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    while True:
        data = input(
            "If you want to exit, type \'out\'. Type dataset you want to implement, \'SVMguide1\' or \'Landsat\': ")

        if data == 'out':
            print('Finished')
            break
        if data not in ['out', 'SVMguide1', 'Landsat']:
            print('Your answer should \'out\' or \'SVMguide1\' or \'Landsat\'')

        if data == 'Landsat':
            print('Now training using Landsat data is in progress.....')
            df = pd.read_csv('Landsat.tst', sep=' ', header=None)
            df = df.astype('float')
            X_data = df.drop(36, axis=1)

            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(X_data)
            scaled = pd.DataFrame(scaler.transform(X_data))

            scaled['class'] = df[36]
            df = scaled
            classes = df['class'].unique()
            normal = [1, 2, 3]
            out = [4, 5, 7]
            indice = []
            outIndice = []
            for c in normal:
                indice += df['class'].index[df['class'] == c].tolist()

            for c in out:
                outIndice += df['class'].index[df['class'] == c].tolist()

            indice = np.sort(indice)
            outIndice = np.sort(outIndice)

            scores_Landsat_macro = []
            scores_Landsat_micro = []
            scores_Landsat_weighted = []

            # for iter_ in range(10):
            # print('iter : ', iter_)

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

            model1 = PseudoSamples(target_train1, 3)
            opt_comb1 = model1.search_optimal_hyperparameters()

            clf1 = OneClassSVM(nu=opt_comb1[0], gamma=opt_comb1[1]).fit(target_train1)
            # classification time measurement

            y_pred1 = []
            times = []
            for i in range(len(testset1)):
                start = time.time()
                clf1.predict([testset1.iloc[i, :]])
                times.append(time.time() - start)
                y_pred1.append(clf1.predict([testset1.iloc[i, :]]).item())

            # y_pred1 = clf1.predict(testset1)

            score1_weighted = f1_score(y_true1, y_pred1, average='weighted')
            time_for_clf = np.mean(times)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true1, y_pred1, average='weighted')

            print('==========Test result for Landsat data==========', '\n')
            print('optimal hyperparamters : nu = ', opt_comb1[0], ', sigma = ', opt_comb1[1], '\n')
            print('f1-score = ', f1, '\n')
            print('precision = ', precision, '\n')
            print('recall = ', recall, '\n')
            print('time for classificaiton : ', time_for_clf, 'sec')
            print('==========Test finished==========\n')
            ans = input('If you want to check classification result for 20 samples randomly chosen, type \'check\' : ')
            if ans == 'check':
                y_pred, y_true = zip(*random.sample(list(zip(y_pred1, y_true1)), 20))
                print('Model precition for 20 randomly chosen test data : ', y_pred)
                print('True class for 20 randomly chosen test data : ', y_true)

        if data == 'SVMguide1':
            print('Now training using SVMguide1 data is in progress.....')
            df = pd.read_csv('SVMguide1.txt', header=None, sep=' ')

            classes = df[0]
            df = df.drop(0, axis=1)
            val = []
            for j in range(df.shape[0]):
                for i in range(1, 5):
                    val.append(str(df[i][j]).split(':')[1])

            val = np.array(val, dtype=float)
            val = val.reshape(df.shape)
            df = pd.DataFrame(val)
            df['class'] = classes

            X_data = df.drop('class', axis=1)
            X_data = X_data.astype('float')
            scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
            scaler.fit(X_data)
            scaled = pd.DataFrame(scaler.transform(X_data))
            scaled['class'] = df['class']
            df = scaled
            classes = df['class'].unique()
            scores_SVMguide1_weighted = []

            target1 = df[df['class'] == 1]
            target_train1 = target1.sample(frac=0.7)
            target_val1 = target1.drop(target_train1.index)
            target_train1 = target_train1.reset_index(drop=True)
            target_val1 = target_val1.reset_index(drop=True)
            outliers1 = df[df['class'] != 1].reset_index(drop=True)
            testset1 = pd.concat([target_val1, outliers1]).reset_index(drop=True)

            y_true1 = [1 if i == 1 else -1 for i in testset1['class']]
            target_train1 = target_train1.drop('class', axis=1)
            testset1 = testset1.drop('class', axis=1)

            model1 = PseudoSamples(target_train1, 3)
            opt_comb1 = model1.search_optimal_hyperparameters()
            clf1 = OneClassSVM(nu=opt_comb1[0], gamma=opt_comb1[1]).fit(target_train1)

            y_pred1 = []
            times = []
            for i in range(len(testset1)):
                start = time.time()
                clf1.predict([testset1.iloc[i, :]])
                times.append(time.time() - start)
                y_pred1.append(clf1.predict([testset1.iloc[i, :]]).item())
            time_for_clf = np.mean(times)

            precision, recall, f1, _ = precision_recall_fscore_support(y_true1, y_pred1, average='weighted')

            print('==========Test result for SVMguide1 data==========', '\n')
            print('optimal hyperparamters : nu = ', opt_comb1[0], ', sigma = ', opt_comb1[1], '\n')
            print('f1-score = ', f1, '\n')
            print('precision = ', precision, '\n')
            print('recall = ', recall, '\n')
            print('time for classificaiton : ', time_for_clf, 'sec')
            print('==========Test finished==========')
            ans = input('If you want to check classification result for 20 samples randomly chosen , type \'check\' : ')
            if ans == 'check':
                y_pred, y_true = zip(*random.sample(list(zip(y_pred1, y_true1)), 20))
                print('Model precition for 20 randomly chosen test data : ', y_pred)
                print('True class for 20 randomly chosen test data : ', y_true)


