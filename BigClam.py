import numpy as np
import random
import operator
import time
import sys
sumFV = []


def getConductance(adjlst, vset, m):
    cut = 0
    vol = 0
    for v in vset:
        for nghr in adjlst[v]:
            vol += 1
            if nghr not in vset:
                cut += 1
    return cut/float(min(vol,m-vol))

def localMinNeig(G):
    maps = {}
    adjlst = G.list
    m = G.m
    for v in adjlst:
        vset = [e for e in adjlst[v]]
        vset.append(v)
        maps[v] = getConductance(adjlst,vset,m)
    return sorted(maps.items(),key=operator.itemgetter(1))



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

def rndInit(G, k):
    global sumFV
    sumFV = [0 for i in range(k)]
    vertex = [i for i in range(G.n)]
    FMap = {}
    for i in range(G.n):
        FMap[i] = dict()
    for v in vertex:
        mem = len(G.list[v])
        if mem > 10:
            mem = 10
        for c in range(mem):
            i = random.randrange(k)
            val = random.random()
            addCom(FMap, v, i, val)

    for c in range(len(sumFV)):
        if not sumFV[c]:
            node = random.randrange(G.n)
            val = random.random()
            addCom(FMap, node, c, val)

    return FMap

def commInit(G, k):
    global sumFV
    sumFV = [0 for i in range(k)]
    FMap = {}
    for i in range(G.n):
        FMap[i] = dict()
    lists = localMinNeig(G)
    vertexs = set(G.vertex)
    cnt = 0
    for vt, val in lists:
        if vt not in vertexs:
            continue
        vset = G.list[vt]
        vset.append(vt)
        if cnt == k:
            break
        for v in vset:
            if v in vertexs:
                vertexs.remove(v)
            addCom(FMap,v,cnt,1)
        cnt += 1

    # random assign some user for no-member community
    while cnt < k:
        for i in range(10):
            v = random.sample(G,vertexs,1)
            addCom(FMap, v, cnt, 1)
        cnt += 1

    return FMap

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


def gradientRow(G,FMap,node,cidSet,w,epsilon):
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
        if not getCom(FMap,node,cid) and val < 0:
            continue
        if abs(val) < 0.0001:
            continue
        if val < -10:
            val = -10
        elif val > 10:
            val = 10
        GradU[cid] = val
    return GradU


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

def trainByList(G, k, w, epsilon, alpha, beta, theshold, maxIter):
    # F init by local minimal neighborhood
    begin = time.time()
    FMap = rndInit(G, k)
    print("init:{}s".format(time.time()-begin))
    # F init by random init
    # F = rndInit(G, F, k, set(G.vertex))

    adjlst = G.list

    vertex = [i for i in range(G.n)]
    iter = 0
    prevL = -sys.maxsize
    curL = 0
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
        print("iter:{} likelihood:{} time:{}s".format(iter,curL,time.time()-begin))
        if abs((curL-prevL)/prevL) <= theshold:
            break
        else:
            prevL = curL
    return FMap


def bigClam(G, k, alpha=0.05, beta=0.3, theshold=0.0001,maxIter=1000):
    epsilon = 1.0/G.n  # background edge propability in sec. 4
    w = 1
    delta = np.sqrt(-np.log(1 - epsilon))  # threshold to determine user-community edge
    N = G.n

    F = trainByList(G, k, w, epsilon, alpha, beta, theshold, maxIter)

    C = []
    for j in range(k):
        c = []
        for i in range(N):
            if F[i][j] >= delta:
                c.append(i)
        C.append(c)
    return C