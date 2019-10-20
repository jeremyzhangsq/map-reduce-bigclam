import numpy as np
from Omega import Omega
class Graph:
    def __init__(self, filename1, filename2):
        self.readNetwork(filename1)
        self.readCommunity(filename2)


    def readNetwork(self, filename):
        file = open(filename,"r")
        n,m = file.readline().rstrip("\n").split("\t")
        self.n = int(n)
        self.m = 2*int(m)
        self.matrix = np.zeros((self.n, self.n), dtype=np.int8)
        self.list = {}
        for i in range(int(m)):
            s,t = file.readline().rstrip("\n").split("\t")
            s = int(s)
            t = int(t)
            self.matrix[s, t] = 1
            self.matrix[t, s] = 1
            if s not in self.list:
                self.list[s] = [t]
            else:
                self.list[s].append(t)
            if t not in self.list:
                self.list[t] = [s]
            else:
                self.list[t].append(s)
        self.vertex = self.list.keys()
        file.close()

    def readCommunity(self, filename):
        file = open(filename,"r")
        self.community = {}
        cnt = 0
        for line in file.readlines():
            each = line.rstrip("\n").split("\t")
            each = [int(a) for a in each]
            self.community[cnt]=each
            cnt += 1
        file.close()

def outputCommunity(community,file):
    out = open(file,"w")
    for item in community:
        s = ""
        for each in item[:-1]:
            s += str(each) + "\t"
        s += str(item[-1]) + "\n"
        out.write(s)
    out.close()

def f1score(truth, train):
    pass

def omegaIndex(truth, train):
    omega = Omega(train,truth)
    return omega.omega_score


def accuracy(truth, train):
    pass

def writeMetrics(filename, dic):
    pass