import numpy as np

class Graph:
    def __init__(self, filename1, filename2):
        self.readNetwork(filename1)
        self.readCommunity(filename2)


    def readNetwork(self, filename):
        file = open(filename,"r")
        n,m = file.readline().rstrip("\n").split("\t")
        self.n = int(n)
        self.m = int(m)
        self.matrix = np.zeros((self.n, self.n), dtype=np.int8)
        self.list = {}
        for i in range(self.m):
            s,t = file.readline().rstrip("\n").split("\t")
            s = int(s)
            t = int(t)
            self.matrix[s, t] = 1
            if s not in self.list:
                self.list[s] = [t]
            else:
                self.list[s].append(t)
        self.vertex = self.list.keys()
        file.close()

    def readCommunity(self, filename):
        file = open(filename,"r")
        self.community = []
        for line in file.readlines():
            each = line.rstrip("\n").split("\t")
            each = [int(a) for a in each]
            self.community.append(each)
        file.close()

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