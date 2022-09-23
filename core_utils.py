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
np.seterr(divide='ignore', invalid='ignore')


class OptimumHyperparamFinder:
    def __init__(self, data, generation_type='sampling'):
        """
        :param data: 최적의 하이퍼 파라미터를 찾기 위한 데이터셋 입니다. outlier가 없는 참(normal) 데이터로 이루어진 pandas dataframe이며,
        m차원 데이터, n개의 샘플 수를 가정할 시 컬럼은 (n, m) 형태의 데이터프레임 스키마를 반드시 맞춰야 합니다.

        :param generation_type: outlier 생성 방법이며, "replace"와 "sampling" 중 "sampling"을 기본 값으로 합니다.
        """

        # test_num : number of acceptance rejection test by uniform number
        self.n_sample = data.shape[0]
        self.dim = data.shape[1]
        self.data = copy.deepcopy(data)
        self.n_neighbors = int(5 * np.log(len(data)))
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(data)
        self.dists = self.nbrs.kneighbors(data)[0]
        self.epsilon = np.mean(np.max(self.dists, axis=1))
        self.test_num = int(2 * np.sqrt(self.dim))
        self.generation_type = generation_type

    def _get_neighbors(self, index_of_target):

        point_tree = cKDTree(self.data)
        arr = np.copy(self.data.iloc[index_of_target, :], order='C')
        idx = point_tree.query_ball_point(arr, self.epsilon)
        point_set = self.data.loc[idx, :]
        target = [True if i == index_of_target else False for i in idx]
        point_set['target'] = target

        return idx, point_set

    def _get_features(self):

        # f1 : epsilon 내에 잡힌 neighbor target의 갯수; volume 추정량
        # f2 : neighbor target들의 중심과 target point 의 거리
        # f3 : outlier 확정 여부
        f1 = []
        f2 = []
        f3 = []
        target_candidates = []
        outlier_candidates = []
        for i in range(len(self.data)):
            idx, point_set = self._get_neighbors(i)
            target_coord = np.squeeze(point_set[point_set['target'] == True].values)[0:self.dim]

            f1_ = len(idx)
            f2_ = np.nan if len(idx) == 1 else distance.euclidean(target_coord, np.mean(point_set.iloc[:, 0:self.dim]))
            f3_ = 'outlier' if len(idx) == 1 else 'normal'
            if len(idx) != 1:
                # generation directions with sampling with replace
                pseudo_outlier = []
                for _ in range(2 * self.test_num):
                    if self.generation_type == 'replace':
                        p_o = target_coord - np.mean(point_set.sample(frac=1, replace=True).iloc[:, 0:self.dim])
                    elif self.generation_type == 'sampling':
                        p_o = target_coord - np.mean(point_set.sample(frac=0.8).iloc[:, 0:self.dim])
                    p_o = self.epsilon * (p_o / np.linalg.norm(p_o)) + target_coord
                    pseudo_outlier.append(p_o)
            else:
                pseudo_outlier = [target_coord]

            '''
            if len(idx) != 1:
                pseudo_target = []
                for _ in range(2*self.test_num):
                    try:
                        #density gradient만 살짝 바꾸어 가고 neighbor는 그대로 둔다
                        p_t = np.array(np.mean(point_set.sample(frac=1, replace=True).iloc[:, 0:self.dim]) - target_coord, dtype=float)
                        p_t = p_t / np.linalg.norm(p_t)
                        neighbors = np.array(point_set.iloc[:, 0:self.dim] - target_coord, dtype=float)
                        projections = np.matmul(neighbors, p_t)
                        p_t = p_t * min(projections[projections > 0]) + target_coord
                    except ValueError:
                        p_t = [np.nan] * self.dim
                    pseudo_target.append(p_t)
            else:
                pseudo_target = [[np.nan] * self.dim]
            '''
            if len(idx) != 1:
                try:
                    pseudo_target = np.array(np.mean(point_set.iloc[:, 0:self.dim]) - target_coord, dtype=float)
                    pseudo_target = pseudo_target / np.linalg.norm(pseudo_target)
                    neighbors = np.array(point_set.iloc[:, 0:self.dim] - target_coord, dtype=float)
                    projections = np.matmul(neighbors, pseudo_target)
                    pseudo_target = pseudo_target * min(projections[projections > 0]) + target_coord
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
        # prob = np.array([1 if np.isnan(prob[i]) else (prob[i] - min_) / (max_ - min_) for i in range(len(prob))])

        return f1, f2, f3, outlier_candidates, target_candidates

    def _get_pseudo_points(self):

        f1, f2, f3, outlier_candidates, target_candidates = self._get_features()

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
        prob[less_idx] = less ** 2
        prob[over_idx] = np.sqrt(over)

        self.data['prob'] = prob

        pseudo_outliers = np.empty((0, self.dim), dtype=float)
        pseudo_targets = np.empty((0, self.dim), dtype=float)

        for i in range(self.test_num):
            unif = np.random.uniform(size=len(prob))
            # 사용된 outlier,target은 list에서 지운다
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
                        accepted_outlier = np.expand_dims(outlier_candidates[j][r - 1], 0)
                        outlier_candidates[j].pop(r - 1)
                    else:
                        accepted_outlier = outlier_candidates[j][0].reshape(1, self.dim)
                    pseudo_outliers = np.append(pseudo_outliers, accepted_outlier, axis=0)

            # accepted_targets = target_candidates[unif>prob]
            # pseudo_targets = np.append(pseudo_targets, accepted_targets, axis=0)

        nearest_mean = np.mean(
            [np.sort(np.sqrt(np.sum((i - np.array(self.data)) ** 2, axis=1)))[3] for i in np.array(self.data)])
        point_tree = cKDTree(self.data.iloc[:, 0:self.dim])

        pseudo_outliers_to_remove = []
        for i, o in enumerate(pseudo_outliers):
            idx = point_tree.query_ball_point(o, nearest_mean)
            if len(idx) > 1:
                pseudo_outliers_to_remove.append(i)
            if sum(np.isnan(list(o))) == self.dim:
                pseudo_outliers_to_remove.append(i)

        pseudo_targets_to_remove = []
        for i, t in enumerate(pseudo_targets):
            if sum(np.isnan(list(t))) == self.dim:
                pseudo_targets_to_remove.append(i)

        pseudo_outliers = np.delete(pseudo_outliers, pseudo_outliers_to_remove, 0)

        pseudo_targets = np.delete(pseudo_targets, pseudo_targets_to_remove, 0)

        return np.array(pseudo_outliers, dtype='float'), np.array(pseudo_targets, dtype='float'), self.data

    def search_optimal_hyperparameters(self):
        """

        :return: OCSVM 모델을 적합하기 위해 현재 주어진 데이터셋(self.data)에 최적화 된 하이퍼파라미터 (nu, gamma)를 return 합니다.
        """

        pseudo_outliers, pseudo_targets, data_with_prob = self._get_pseudo_points()

        sigma = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10 ** 1, 10 ** 2, 10 ** 3,
                 10 ** 4]
        nu = [0.01, 0.05, 0.1, 0.15, 0.2]
        comb_list = [(i, j) for i in nu for j in sigma]
        errs = []
        for comb in comb_list:
            nu_, gamma_ = comb
            clf = OneClassSVM( nu=nu_, gamma=gamma_).fit(data_with_prob.drop('prob', axis=1))
            err_o = sum(clf.predict(pseudo_outliers) == 1) / len(pseudo_outliers)
            err_t = sum(clf.predict(pseudo_targets) == -1) / len(pseudo_targets)
            err = (err_o + err_t) / 2
            errs.append(err)

        for comb, err in zip(comb_list, errs):
            if err == min(errs):
                opt_comb = comb

        # clf = OneClassSVM(nu=opt_comb[0], gamma=opt_comb[1]).fit(data_with_prob.drop('prob', axis=1))

        # if self.dim == 2:
        #     plt.scatter(data_with_prob.iloc[:, 0], data_with_prob.iloc[:, 1], c=data_with_prob['prob'], cmap='Blues',
        #                 label='original_data', s=20)
        #     plt.scatter(np.array(pseudo_outliers)[:, 0], np.array(pseudo_outliers)[:, 1], color='red', marker='x',
        #                 label='pseudo_outliers', s=20)
        #     plt.scatter(np.array(pseudo_targets)[:, 0], np.array(pseudo_targets)[:, 1], color='green', marker='*',
        #                 label='pseudo_targets', s=20)
        #
        #     plt.title('optimal hyperparameters : nu=' + str(opt_comb[0]) + ', sigma=' + str(opt_comb[1]), size=20)
        #
        #     h = .02  # step size in the mesh
        #     # create a mesh to plot in
        #     x_min, x_max = pseudo_outliers[:, 0].min() - 0.1, pseudo_outliers[:, 0].max() + 0.1
        #     y_min, y_max = pseudo_outliers[:, 1].min() - 0.1, pseudo_outliers[:, 1].max() + 0.1
        #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                          np.arange(y_min, y_max, h))
        #
        #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #
        #     # Put the result into a color plot
        #     Z = Z.reshape(xx.shape)
        #     plt.contour(xx, yy, Z, colors='gold', linewidths=5)  # cmap=plt.cm.Paired, color='purple')
        #     plt.legend(loc=2, fontsize='large')
        return opt_comb

