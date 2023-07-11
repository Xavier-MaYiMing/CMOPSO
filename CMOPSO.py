#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/18 16:29
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : CMOPSO.py
# @Statement : A competitive mechanism based multi-objective particle swarm optimizer (CMOPSO)
# @Reference : Zhang X, Zheng X, Cheng R, et al. A competitive mechanism based multi-objective particle swarm optimizer with fast convergence[J]. Information Sciences, 2018, 427: 63-76.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < pm / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def nd_cd_sort(pop, objs, gamma):
    # sort the population according to the Pareto rank and crowding distance
    pfs, rank = nd_sort(objs)  # Pareto fronts and Pareto ranks
    cd = crowding_distance(objs, pfs)  # crowding distance
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    leaders = np.zeros((gamma, pop.shape[1]))
    leaders_objs = np.zeros((gamma, objs.shape[1]))
    for i in range(gamma):
        leaders[i] = temp_list[i][0]
        leaders_objs[i] = temp_list[i][1]
    return leaders, leaders_objs


def truncation(objs, k):
    # truncate k particles with the minimum distance from the other particles
    delete = np.full(objs.shape[0], False)
    fmax = np.max(objs, axis=0)
    fmin = np.min(objs, axis=0)
    objs = (objs - fmin) / (fmax - fmin)
    sigma = squareform(pdist(objs, metric='euclidean'), force='no', checks=True)
    eye = np.arange(len(sigma))
    sigma[eye, eye] = np.inf
    while np.sum(delete) < k:
        remain = np.where(~delete)[0]
        temp = np.sort(sigma[remain][:, remain])
        delete[remain[np.argmin(temp[:, 0])]] = True
    return delete


def main(npop, iter, lb, ub, pm=1, eta_m=20, gamma=10):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param pm: mutation probability (default = 1)
    :param eta_m: perturbance factor distribution index (default = 20)
    :param gamma: elite particle size (default = 10)
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # dimension
    pos = np.random.uniform(lb, ub, (npop, dim))  # position
    objs = np.array([cal_obj(pos[i]) for i in range(npop)])  # the objectives
    nobj = len(objs[0])  # objective number
    vmax = 0.5 * (ub - lb)  # maximum velocity
    vmin = -vmax  # minimum velocity
    vel = np.random.uniform(vmin, vmax, (npop, dim))  # velocity

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Select leaders
        leaders, leaders_objs = nd_cd_sort(pos, objs, gamma)

        # Step 2.2. Learning
        off_pos = np.zeros((npop, dim))
        off_objs = np.zeros((npop, nobj))
        off_vel = np.zeros((npop, dim))
        for i in range(npop):
            [ind1, ind2] = np.random.choice(gamma, 2, replace=False)
            leader1 = leaders[ind1]
            obj1 = leaders_objs[ind1]
            leader2 = leaders[ind2]
            obj2 = leaders_objs[ind2]
            c1 = np.sum(obj1 * objs[i]) / (np.linalg.norm(obj1) * np.linalg.norm(objs[i]))
            c2 = np.sum(obj2 * objs[i]) / (np.linalg.norm(obj2) * np.linalg.norm(objs[i]))
            c1 = min(max(-1, c1), 1)
            c2 = min(max(-1, c2), 1)
            angle1 = np.rad2deg(np.arccos(c1))
            angle2 = np.rad2deg(np.arccos(c2))
            winner = leader1 if angle1 < angle2 else leader2
            off_vel[i] = np.random.random(dim) * vel[i] + np.random.random(dim) * (winner - pos[i])
            off_vel[i] = np.min((off_vel[i], vmax), axis=0)
            off_vel[i] = np.max((off_vel[i], vmin), axis=0)
            off_pos[i] = pos[i] + off_vel[i]
            off_pos[i] = np.min((off_pos[i], ub), axis=0)
            off_pos[i] = np.max((off_pos[i], lb), axis=0)
            off_objs[i] = cal_obj(off_pos[i])

        # Step 2.3. Polynomial mutation
        off_pos = mutation(off_pos, lb, ub, pm, eta_m)

        # Step 2.4. Environmental selection
        pos = np.concatenate((pos, off_pos), axis=0)
        vel = np.concatenate((vel, off_vel), axis=0)
        objs = np.concatenate((objs, off_objs), axis=0)
        flag = np.full(pos.shape[0], False)
        pfs, rank = nd_sort(objs)
        pf = 1
        length = 0
        while length + len(pfs[pf]) < npop:
            length += len(pfs[pf])
            flag[pfs[pf]] = True
            pf += 1
        next_objs = objs[pfs[pf]]
        delete = truncation(next_objs, length + len(pfs[pf]) - npop)
        flag[np.array(pfs[pf])[~delete]] = True
        pos = pos[flag]
        vel = vel[flag]
        objs = objs[flag]

    # Step 3. Sort the results
    pfs = nd_sort(objs)[0]
    pf = objs[pfs[1]]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of ZDT3')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 200, np.array([0] * 10), np.array([1] * 10))
