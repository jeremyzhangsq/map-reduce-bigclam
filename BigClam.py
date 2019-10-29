import numpy as np
import random
import operator
import time
import sys
import seaborn as sns
import Util
import CPM
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
sumFV = []


def addCom(FMap, nid, cid, val):
    global sumFV
    if cid in FMap[nid]:
        sumFV[cid] -= FMap[nid][cid]
    FMap[nid][cid] = val
    sumFV[cid] += val


def delCom(F, nid, cid):
    global sumFV
    if cid in F[nid]:
        sumFV[cid] -= F[nid][cid]
        del F[nid][cid]


def getCom(F, nid, cid):
    if cid in F[nid]:
        return F[nid][cid]
    else:
        return 0


def dotproduct(Fu, Fv):
    if len(Fu) > len(Fv):
        F1 = Fu
        F2 = Fv
    else:
        F1 = Fv
        F2 = Fu
    dp = 0
    for ele in F1:
        if ele in F2:
            dp += F1[ele]*F2[ele]
    return dp


def Sum(gradU):
    N = 0
    for each in gradU:
        N += gradU[each]
    return N


def norm2(gradU):
    N = 0
    for each in gradU:
        N += gradU[each]**2
    return N


def prediction(Fu, Fv, epsilon):
    dp = np.log(1.0/(1.0-epsilon)) + dotproduct(Fu, Fv)
    return np.exp(-dp)


def getConductance(adjlst, vset, m):
    cut = 0
    vol = 0
    edge = m
    phi = 0
    for v in vset:
        for nghr in adjlst[v]:
            vol += 1
            if nghr not in vset:
                cut += 1
    if vol != edge:
        if 2*vol > edge:
            phi = cut/float(edge-vol)
        elif not vol:
            phi = 0
        else:
            phi = cut/float(vol)
    else:
        if vol == edge:
            phi = 1
    return phi


def localMinNeig(G):
    maps = {}
    adjlst = G.list
    m = G.m
    for v in adjlst:
        if len(adjlst[v]) < 5:
            maps[v] = 1
        else:
            vset = [e for e in adjlst[v]]
            vset.append(v)
            maps[v] = getConductance(adjlst, vset, m)
    return sorted(maps.items(), key=operator.itemgetter(1))


def commInit(G, k):
    global sumFV
    sumFV = [0 for i in range(k)]
    FMap = {}
    for i in range(G.n):
        FMap[i] = dict()
    lists = localMinNeig(G)
    vertexs = set()
    cnt = 0
    for vt, val in lists:
        if vt in vertexs:
            continue
        vset = [i for i in G.list[vt]]
        vset.append(vt)
        for v in vset:
            addCom(FMap, v, cnt, 1)
            vertexs.add(v)
        cnt += 1
        if cnt >= k:
            break
    if k > cnt:
        print("{} communities needed to fill randomly".format(k-cnt))
    # random assign some user for no-member community
    for i in range(k):
        val = sumFV[i]
        if not val:
            for idx in range(10):
                v = random.sample(G.vertex, 1)[0]
                addCom(FMap, v, i, random.random())
    return FMap


def randInit(G, k):
    global sumFV
    sumFV = [0 for i in range(k)]
    FMap = {}
    for i in range(G.n):
        FMap[i] = dict()
    for node in FMap:
        mem = len(G.list[node])
        if mem > 10:
            mem = 10
        for c in range(mem):
            cid = random.randrange(0, k)
            addCom(FMap, node, cid, random.random())

    for i in range(len(sumFV)):
        if not sumFV[i]:
            v = random.sample(G.vertex, 1)[0]
            addCom(FMap, v, i, random.random())
    return FMap


def cpmInit(G, k):
    global sumFV
    sumFV = [0 for i in range(k)]
    FMap = {}
    for i in range(G.n):
        FMap[i] = dict()
    trainComm = CPM.CPM(G, k)
    cnt = 0
    while cnt < k:
        for com in trainComm:
            for user in trainComm[com]:
                v = int(user)
                # todo: change original vertex number
                addCom(FMap, v, com, 1)
            cnt += 1
        if cnt < k:
            break
    if k > cnt:
        print("{} communities needed to fill randomly".format(k - cnt))
    # random assign some user for no-member community
    # for i in range(k):
    #     val = sumFV[i]
    #     if not val:
    #         for idx in range(10):
    #             v = random.sample(G.vertex,1)[0]
    #             addCom(FMap, v, i, random.random())
    return FMap


def get_gradient(g, node, FMap, cid, preV, w):
    val = 0
    for ngh in g:
        if ngh == node:
            continue
        cm = getCom(FMap, ngh, cid)
        val = val + (preV[ngh] * cm / (1-preV[ngh]) + w*cm)
    return val


def gradientRow(G, FMap, node, cidSet, w, epsilon, RegCoef, n_jobs=50):

    preV = {}
    GradU = {}
    for e in G[node]:
        if e == node:
            continue
        preV[e] = prediction(FMap[node], FMap[e], epsilon)
    for cid in cidSet:
        val = 0
        for ngh in G[node]:
            if ngh == node:
                continue
            cm = getCom(FMap, ngh, cid)
            val = val + (preV[ngh] * cm / (1-preV[ngh]) + w*cm)
        # todo: used for holdout set
        # val -= w*(sumFV[cid]-getCom(FMap,node,cid))
        GradU[cid] = val

#     #      multiprocessing
#     G_list = G.list[node]
#     Gs = []
#     lin_range = np.int64(np.linspace(0, len(G_list), n_jobs + 1))
#     for i in range(n_jobs):
#         temp = G_list[lin_range[i]:lin_range[i + 1]]
#         Gs.append(temp)
#     for cid in cidSet:
#         with Pool(n_jobs) as p:
#             vals = p.starmap(get_gradient, [(g, node, FMap, cid, preV, w) for g in Gs])
#         GradU[cid] = sum(vals)
#     ################################

    if RegCoef > 0:
        for cid in GradU:
            GradU[cid] -= RegCoef
    if RegCoef < 0:
        for cid in GradU:
            GradU[cid] += 2 * RegCoef*getCom(FMap, node, cid)

    GradV = {}
    for cid in GradU:
        val = GradU[cid]
        if not getCom(FMap, node, cid) and val < 0:
            continue
        if abs(val) < 0.0001:
            continue
        GradV[cid] = val
    for cid in GradV:
        val = GradV[cid]
        if val < -10:
            GradV[cid] = -10
        elif val > 10:
            GradV[cid] = 10

    return GradV


def LikehoodForRow(G, FMap, u, Fu, w, epsilon, RegCoef):
    L = 0
    for ngh in G[u]:
        L += np.log(1-prediction(Fu, FMap[ngh],
                                 epsilon)) + w*dotproduct(Fu, FMap[ngh])
    # todo: used for holdout set
    # for cid in Fu:
    #     L -= w*(sumFV[cid]-getCom(FMap,u,cid))*Fu[cid]

    if RegCoef > 0:
        L -= RegCoef*Sum(Fu)
    else:
        L += RegCoef*norm2(Fu)
    return L


def Likehood(G, FMap, w, epsilon, RegCoef, n_jobs=5):
    L = 0
    for u in FMap:
        L += LikehoodForRow(G, FMap, u, FMap[u], w, epsilon, RegCoef)
    return L


def getStepByLinearSearch(u, G, FMap, deltaV, gradV, w, epsilon, stepAlpha, stepBeta, RegCoef, Maxiter=10):
    stepSize = 1
    initLikehood = LikehoodForRow(G, FMap, u, FMap[u], w, epsilon, RegCoef)
    newmap = {}
    MinVal = 0
    MaxVal = 1000
    for i in range(Maxiter):
        for cid in deltaV:
            newval = getCom(FMap, u, cid)+stepSize*deltaV[cid]
            if newval < MinVal:
                newval = MinVal
            if newval > MaxVal:
                newval = MaxVal
            newmap[cid] = newval
        if LikehoodForRow(G, FMap, u, newmap, w, epsilon, RegCoef) < initLikehood + stepAlpha*stepSize*dotproduct(gradV, deltaV):
            stepSize *= stepBeta
        else:
            break
        if i == Maxiter-1:
            stepSize = 0
            break
    return stepSize


def getCommunity(F, delta):
    C = {}
    for i in range(len(sumFV)):
        C[i] = list()
    for com in range(len(sumFV)):
        if sumFV[com] < delta:
            continue
        for u in range(len(F)):
            if getCom(F, u, com) > delta:
                C[com].append(u)
    result = {}
    for i in C:
        if C[i]:
            result[i] = C[i]
    return result


def get_vertex(vertex, adjlst, FMap, alpha, beta, w, epsilon, RegCoef):
    for person in vertex:
        cset = set()
        todel = set()
        for ngh in adjlst[person]:
            cset = cset.union(set(FMap[ngh].keys()))
        for each in FMap[person]:
            if each not in cset:
                todel.add(each)
        for each in todel:
            delCom(FMap, person, each)
        if not len(cset):
            continue
        gradv = gradientRow(adjlst, FMap, person, cset, w, epsilon, RegCoef)
        if norm2(gradv) < 1e-4:
            continue
        learnRate = getStepByLinearSearch(
            person, adjlst, FMap, gradv, gradv, w, epsilon, alpha, beta, RegCoef)
        if not learnRate:
            continue
        for cid in gradv:
            change = learnRate*gradv[cid]
            newFuc = getCom(FMap, person, cid)+change
            if newFuc <= 0:
                delCom(FMap, person, cid)
            else:
                addCom(FMap, person, cid, newFuc)
    return FMap


def trainByList(G, truth, k, delta, w, epsilon, alpha, beta, theshold, maxIter, RegCoef, n_jobs=10):
    # F init by local minimal neighborhood
    begin = time.time()
    FMap = commInit(G, k)
    # FMap = randInit(G, k)
    # FMap = cpmInit(G, k)
    comm = getCommunity(FMap, delta)
    f1 = Util.f1score(truth, comm)
    print("init:{}s f1score:{}".format(time.time() - begin, f1))
    adjlst = G.list
    vertex = [i for i in range(G.n)]
    iter = 0
    prevIter = 0
    prevL = -1.79769e+308
    curL = 0
    f1score = []
    xiter = []
    begin = time.time()
    while iter < maxIter:
        itertime = time.time()
        random.shuffle(vertex)
        reg = RegCoef
        iter += 1

        if n_jobs > 0:
            vertexs = []
            lin_range = np.int64(np.linspace(0, len(vertex), n_jobs + 1))
            for i in range(n_jobs):
                temp = vertex[lin_range[i]:lin_range[i + 1]]
                vertexs.append(temp)
            with Pool(n_jobs) as p:
                FMaps = p.starmap(get_vertex, [
                    (v, adjlst, FMap, alpha, beta, w, epsilon, RegCoef) for v in vertexs])
            FMap = {}
            for f in FMaps:
                FMap.update(f)
        else:
            for person in vertex:
                cset = set()
                todel = set()
                for ngh in adjlst[person]:
                    cset = cset.union(set(FMap[ngh].keys()))
                for each in FMap[person]:
                    if each not in cset:
                        todel.add(each)
                for each in todel:
                    delCom(FMap, person, each)
                if not len(cset):
                    continue
                gradv = gradientRow(adjlst, FMap, person, cset, w, epsilon, RegCoef)
                if norm2(gradv) < 1e-4:
                    continue
                learnRate = getStepByLinearSearch(
                    person, adjlst, FMap, gradv, gradv, w, epsilon, alpha, beta, RegCoef)
                if not learnRate:
                    continue
                for cid in gradv:
                    change = learnRate*gradv[cid]
                    newFuc = getCom(FMap, person, cid)+change
                    if newFuc <= 0:
                        delCom(FMap, person, cid)
                    else:
                        addCom(FMap, person, cid, newFuc)

        curL = Likehood(G.list, FMap, w, epsilon, RegCoef)
        comm = getCommunity(FMap, delta)
        fname = "./visual/bigclam_output_{}.txt".format(iter)
        # Util.outputCmty(comm,fname)
        f1 = Util.f1score(truth, comm)
        avgnum = Util.avgCommNum(comm)
        f1score.append(f1)
        xiter.append(iter)
        print("iter:{} likelihood:{:.3f} delta:{:.4f} time:{:.3f}s f1score:{:.3f} avgcomm:{}".format(
            iter, curL, abs((curL - prevL) / prevL), time.time() - itertime, f1, avgnum))
        if (iter-1) % 5 == 0:
            llval = []
            for item in FMap:
                for val in FMap[item]:
                    llval.append(FMap[item][val])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            sns.distplot(llval)
            plt.xlim([0, 400])
            ax.set_yscale('log')
            ax.set_ylim(ymax=1)
            ax.set_ylim(ymin=10**-6)
            plt.title("iteration:{}".format(iter-1))
            plt.savefig("./log/f_kde{}.png".format(iter-1))
            plt.close()

        if abs((curL-prevL)/prevL) <= theshold:
            break
        else:
            prevL = curL
    plt.figure()
    plt.plot(xiter, f1score)
    # plt.ylim([0,1])
    plt.savefig("f1score_iter.png")
    plt.close()
    return FMap


def bigClam(G, truth, k, delta, alpha=0.05, beta=0.3, theshold=0.01, maxIter=1000, RegCoef=1):
    epsilon = 10**(-8)  # background edge propability in sec. 4
    w = 1
    n_jobs = 10
    F = trainByList(G, truth, k, delta, w, epsilon, alpha,
                    beta, theshold, maxIter, RegCoef, n_jobs)
    C = getCommunity(F, delta)
    return C
