import numpy as np
from Omega import Omega
import re
import sys
from pyspark import SparkConf, SparkContext
from multiprocessing import Pool

class Graph:
    def __init__(self, filename1, filename2):
        self.readNetwork(filename1)
#         self.readNetwork_mapreduce(filename1)
        self.readCommunity(filename2)

    def readNetwork(self, filename):
        file = open(filename, "r")
        n, m = file.readline().rstrip("\n").split("\t")
        self.n = int(n)
        self.m = 2*int(m)
        self.matrix = np.zeros((self.n, self.n), dtype=np.int8)
        self.list = {}
        for i in range(int(m)):
            s, t = file.readline().rstrip("\n").split("\t")
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

    def readNetwork_mapreduce(self, filename):
        # initialize the environment
        conf = SparkConf() 
        sc = SparkContext(conf=conf)
        
        file = open(filename,"r")
        n,m = file.readline().rstrip("\n").split("\t")
        n = int(n)
        m = int(m)
        self.n = n
        self.m = 2 * m
        file.close()
        
        lines = sc.textFile(filename) # read text
        header = lines.first()
        header = sc.parallelize([header])
        lines = lines.subtract(header)
        pairs = lines.map(lambda l: re.split('\t', l)) # flatten + map
        pairs = pairs.map(lambda w: (w[0], [w[1]]))
        
        self.matrix = np.zeros((self.n, self.n), dtype=np.int8)
        self.list = {}
        for item in pairs.collect():
            s = item[0]
            for t in item[1]:
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

def outputCommunity(metrics,community,file):
    '''
    outputfile format:
        algorithm\n
        readtime\n
        executiontime\n
        f1score\n
        omega-index\n
        accuracy\n
        lines of communities (users inside a community is delimited by \t)
    '''
    out = open(file,"w")
    out.write("{}\n".format(metrics["algorithm"]))
    out.write("{}\n".format(metrics["readTime"]))
    out.write("{}\n".format(metrics["execTime"]))
    out.write("{}\n".format(metrics["f1score"]))
    # out.write("{}\n".format(metrics["omgIdx"]))
    out.write("{}\n".format(metrics["accuracy"]))
    for i in community:
        s = ""
        item = community[i]
        for each in item[:-1]:
            s += str(each) + "\t"
        s += str(item[-1]) + "\n"
        out.write(s)
    out.close()

def outputCmty(community,file):
    '''
    outputfile format:
        lines of communities (users inside a community is delimited by \t)
    '''
    out = open(file,"w")
    # out.write("{}\n".format(metrics["algorithm"]))
    # out.write("{}\n".format(metrics["readTime"]))
    # out.write("{}\n".format(metrics["execTime"]))
    # out.write("{}\n".format(metrics["f1score"]))
    # out.write("{}\n".format(metrics["omgIdx"]))
    # out.write("{}\n".format(metrics["accuracy"]))
    for i in community:
        s = ""
        item = community[i]
        for each in item[:-1]:
            s += str(each) + "\t"
        s += str(item[-1]) + "\n"
        out.write(s)
    out.close()

def F1(com1,com2):
    correctly_classified = list(set(com1).intersection(set(com2)))
    precision = len(correctly_classified) / float(len((com1)))
    recall = len(correctly_classified) / float(len(com2))
    if precision != 0 and recall != 0:
        Fscore = 2* precision * recall / float(precision + recall)
    else:
        Fscore = 0
    return Fscore
def bestMatch(one, all):
    maxf1 = 0
    for each in all:
        com2 = all[each]
        score = F1(one,com2)
        if not score:
            continue
        if score>maxf1:
            maxf1 = score
        if maxf1==1:
            return maxf1
    return maxf1

def avgCommNum(realComm):
    userCom = {}
    for com in realComm:
        for user in realComm[com]:
            if user not in userCom:
                userCom[user] = [com]
            else:
                userCom[user].append(com)
    avgcom = 0
    for u in userCom:
        avgcom += len(userCom[u])
    avgcom /= float(len(userCom))
    return avgcom
# def f1score(truth, train):
#     """
#     Quote from WSDM12: We define F1 score to be the average of the F1-score of the best matching ground-truth community
#     to each detected community, and the F1-score of the best-matching detected community to each ground-truth community
#     """
#     truthscore = 0
#     trainscore = 0
#     truthNum = len(truth)
#     trainNum = len(train)
#     for i in truth:
#         truthcom = truth[i]
#         truthscore += bestMatch(truthcom,train)
#     for j in train:
#         traincom = train[j]
#         trainscore += bestMatch(traincom,truth)
#     return 0.5*(trainscore/float(trainNum)+truthscore/float(truthNum))

def get_f1_score_truth(truth, train):
    truthscore = 0
    for i in truth:
        truthcom = truth[i]
        truthscore += bestMatch(truthcom,train)
    return truthscore

def get_f1_score_train(truth, train):
    trainscore = 0
    for j in train:
        traincom = train[j]
        trainscore += bestMatch(traincom,truth)
    return trainscore

def f1score(truth, train, n_jobs=10):
    """
    Quote from WSDM12: We define F1 score to be the average of the F1-score of the best matching ground-truth community
    to each detected community, and the F1-score of the best-matching detected community to each ground-truth community
    """
    truthscore = 0
    trainscore = 0
    truthNum = len(truth)
    trainNum = len(train)

    truth_list = list(truth.items())
    truths = []
    lin_range = np.int64(np.linspace(0, truthNum, n_jobs + 1))
    for i in range(n_jobs):
        temp = truth_list[lin_range[i]:lin_range[i + 1]]
        temp = {item[0]: item[1] for item in temp}
        truths.append(temp)
    with Pool(n_jobs) as p:
        scores = p.starmap(get_f1_score_truth, [(t, train) for t in truths])
    truthscore = sum(scores)

    train_list = list(train.items())
    trains = []
    lin_range = np.int64(np.linspace(0, trainNum, n_jobs + 1))
    for i in range(n_jobs):
        temp = train_list[lin_range[i]:lin_range[i + 1]]
        temp = {item[0]: item[1] for item in temp}
        trains.append(temp)
    with Pool(n_jobs) as p:
        scores = p.starmap(get_f1_score_truth, [(truth, t) for t in trains])
    trainscore = sum(scores)

    return 0.5*(trainscore/float(trainNum)+truthscore/float(truthNum))


def omegaIndex(truth, train):
    omega = Omega(train,truth)
    return omega.omega_score


def accuracy(truth, train):
    return 1-abs(len(truth)-len(train))/float(2*len(truth))
