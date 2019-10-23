import numpy as np
import random
import operator
import time
import sys
import seaborn as sns
import Util
import matplotlib.pyplot as plt
sumFV = []

def addCom(FMap, nid, cid, val):
    global sumFV
    if cid in FMap[nid]:
        sumFV[cid] -= FMap[nid][cid]
    FMap[nid][cid] = val
    sumFV[cid] += val

def delCom(F,nid,cid):
    global sumFV
    if cid in F[nid]:
        sumFV[cid] -= F[nid][cid]
        del F[nid][cid]

def getCom(F,nid,cid):
    if cid in F[nid]:
        return F[nid][cid]
    else:
        return 0

def dotproduct(Fu,Fv):
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

def norm2(gradU):
    N = 0
    for each in gradU:
        N += gradU[each]**2
    return N

def prediction(Fu,Fv,epsilon):
    dp = np.log(1.0/(1.0-epsilon)) + dotproduct(Fu,Fv)
    return np.exp(-dp)


def getConductance(adjlst, vset, m):
    cut = 0
    vol = 0
    edge = 2*m if m >= 0 else m
    for v in vset:
        for nghr in adjlst[v]:
            vol += 1
            if nghr not in vset:
                cut += 1
    if vol!=edge:
        if 2*vol > edge:
            return cut/float(edge-vol)
        elif not vol:
            return 0
        else:
            return cut/float(vol)
    else:
        return 1

def localMinNeig(G):
    maps = {}
    adjlst = G.list
    m = G.m
    for v in adjlst:
        if len(adjlst[v])<5:
            maps[v] = 1
        else:
            vset = [e for e in adjlst[v]]
            vset.append(v)
            maps[v] = getConductance(adjlst,vset,m)
    return sorted(maps.items(),key=operator.itemgetter(1))


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
        vertexs.add(vt)
        addCom(FMap, vt, cnt, 1)
        vset = G.list[vt]
        vset.append(vt)
        if cnt == k:
            break
        for v in vset:
            addCom(FMap,v,cnt,1)
            vertexs.add(v)
        cnt += 1
        if cnt >= k:
            break

    # random assign some user for no-member community
    for i in range(k):
        val = sumFV[i]
        if not val:
            for idx in range(10):
                v = random.sample(G.vertex,1)
                addCom(FMap, v, i, 1)

    return FMap



def gradientRow(G,FMap,node,cidSet,w,epsilon, RegCoef=10):
    preV = {}
    GradU = {}
    for e in G.list[node]:
        preV[e] = prediction(FMap[node],FMap[e],epsilon)

    for cid in cidSet:
        val = 0
        for ngh in G.list[node]:
            cm = getCom(FMap,ngh,cid)
            val += preV[ngh] * cm / (1-preV[ngh])+ w*cm
        val -= w*(sumFV[cid]-getCom(FMap,node,cid))
        GradU[cid] = val

    if RegCoef > 0:
        for cid in GradU:
            GradU[cid] -=RegCoef
    if RegCoef < 0:
        for cid in GradU:
            GradU[cid] += 2 * RegCoef*getCom(FMap,node,cid)

    GradV = {}
    for cid in GradU:
        val = GradU[cid]
        if not getCom(FMap,node,cid) and val < 0:
            continue
        if abs(val) < 0.0001:
            continue
        if val < -10:
            val = -10
        elif val > 10:
            val = 10
        GradV[cid]=val

    return GradV


def LikehoodForRow(G, FMap, u, Fu, w, epsilon):
    L = 0
    for ngh in G.list[u]:
        L += np.log(1-prediction(Fu,FMap[ngh],epsilon)) + w*dotproduct(Fu,FMap[ngh])
    for cid in Fu:
        L -= w*(sumFV[cid]-getCom(FMap,u,cid))*Fu[cid]
    return L

def Likehood(G,FMap,w,epsilon):
    L = 0
    for u in FMap:
        L += LikehoodForRow(G,FMap,u,FMap[u],w,epsilon)
    return L

def getStepByLinearSearch(u, G, FMap, deltaV, gradV, w, epsilon, stepAlpha, stepBeta, Maxiter = 10):
    stepSize = 1
    initLikehood = LikehoodForRow(G, FMap, u, FMap[u],w,epsilon)
    newmap = {}
    MinVal = 0
    MaxVal = 1000
    for i in range(Maxiter):
        for cid in deltaV:
            newval = getCom(FMap,u,cid)+stepSize*deltaV[cid]
            if newval < MinVal:
                newval = MinVal
            if newval > MaxVal:
                newval = MaxVal
            newmap[cid] = newval
        if LikehoodForRow(G, FMap, u, newmap,w, epsilon) < initLikehood +stepAlpha*stepSize*dotproduct(gradV,deltaV):
            stepSize *=stepBeta
        else:
            break
        if i == Maxiter-1:
            stepSize = 0
            break
    return stepSize

def getCommunity(F,delta):
    C = {}
    for user in F:
        for com in F[user]:
            if F[user][com] > delta:
                if com not in C:
                    C[com] = [user]
                else:
                    C[com].append(user)
    return C

def trainByList(G,truth, k, w, epsilon, alpha, beta, theshold, maxIter):
    # F init by local minimal neighborhood
    begin = time.time()
    FMap = commInit(G, k)
    print("init:{}s".format(time.time()-begin))
    adjlst = G.list
    delta = np.sqrt(epsilon)
    vertex = [i for i in range(G.n)]
    iter = 0
    prevL = -sys.maxsize
    curL = 0
    f1score = []
    xiter = []
    begin = time.time()
    while iter < maxIter:
        random.shuffle(vertex)
        for person in vertex:
            cset = set()
            todel = set()
            for ngh in adjlst[person]:
                cset = cset.union(set(FMap[ngh].keys()))
            if not len(cset):
                continue
            for each in FMap[person]:
                if each not in cset:
                    todel.add(each)
            for each in todel:
                delCom(FMap,person,each)

            gradv = gradientRow(G,FMap,person,cset,w,epsilon)
            if norm2(gradv) < 1e-4:
                continue
            learnRate = getStepByLinearSearch(person,G,FMap,gradv,gradv,w, epsilon, alpha,beta)
            if not learnRate:
                continue
            for cid in gradv:
                change = learnRate*gradv[cid]
                newFuc = getCom(FMap,person,cid)+change
                if newFuc <= 0:
                    delCom(FMap,person,cid)
                else:
                    addCom(FMap,person,cid,newFuc)
        iter += 1
        curL = Likehood(G,FMap,w,epsilon)

        comm = getCommunity(FMap,delta)
        f1 = Util.f1score(truth,comm)
        f1score.append(f1)
        xiter.append(iter)
        print("iter:{} likelihood:{:.3f} delta:{:.4f} time:{:.3f}s f1score:{:.3f}".format(iter, curL, abs((curL - prevL) / prevL),
                                                               time.time() - begin,f1))
        if iter%5 == 0:
            llval = []
            for item in FMap:
                for val in FMap[item]:
                    llval.append(FMap[item][val])
            plt.figure()
            sns.distplot(llval)
            plt.ylim([0,5])
            plt.savefig("./log/f_kde{}.png".format(iter))
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



def bigClam(G, truth, k, alpha=0.05, beta=0.3, theshold=0.005,maxIter=1000):
    epsilon = 10**(-8)  # background edge propability in sec. 4
    w = 1
    delta = np.sqrt(epsilon)  # threshold to determine user-community edge
    N = G.n
    F = trainByList(G, truth, k, w, epsilon, alpha, beta, theshold, maxIter)
    C = getCommunity(F,delta)
    return C