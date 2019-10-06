import numpy as np


def readNetwork(filename):
    file = open(filename,"r")
    n,m = file.readline().rstrip("\n").split("\t")
    adjmtx = np.zeros((int(n), int(n)), dtype=np.int8)
    adjlst = {}
    for i in range(int(m)):
        s,t = file.readline().rstrip("\n").split("\t")
        s = int(s)
        t = int(t)
        adjmtx[s,t] = 1
        if s not in adjlst:
            adjlst[s] = [t]
        else:
            adjlst[s].append(t)
    file.close()
    return adjmtx,adjlst

def readCommunity(filename):
    file = open(filename,"r")
    l = []
    for line in file.readlines():
        each = line.rstrip("\n").split("\t")
        each = [int(a) for a in each]
        l.append(each)
    file.close()
    return l

def visualize(community):
    pass

def f1score(truth, train):
    pass

def omegaIndex(truth, train):
    pass

def accuracy(truth, train):
    pass

def writeMetrics(filename, dic):
    pass