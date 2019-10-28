import sys
import time
import Util
from CPM import CPM
import numpy as np
# from LC import LC
from BigClam import bigClam
from NMF import NMF


if __name__ == '__main__':

    # input params
    ufile = sys.argv[1]  # network filename
    cfile = sys.argv[2]  # ground truth community filename
    # select an algorithm. option: bigclam, nmf, lc, cpm
    algorithm = sys.argv[3]
    k = int(sys.argv[4])  # number of community to detect
    output = sys.argv[5]  # output filename

    metrics = dict()

    # load two matrix
    start = time.time()
    graph = Util.Graph(ufile, cfile)
    epsilon = 10 ** (-8)  # background edge propability in sec. 4
    delta = np.sqrt(epsilon)  # threshold to determine user-community edge
    # delta = np.sqrt(2.0 * graph.m / graph.n / graph.n)
    metrics["algorithm"] = algorithm
    metrics["readTime"] = time.time() - start
    realComm = graph.community
    # avarage number of communities per user
    avgnum = Util.avgCommNum(realComm)
    print("Average community per user:{}".format(avgnum))
    # main algorithm
    start = time.time()
    if algorithm == 'bigclam':
        trainComm = bigClam(graph, realComm, k, delta)
    elif algorithm == 'nmf':
        trainComm = NMF(graph, k)
    elif algorithm == 'lc':
        trainComm = LC(graph, k)
    elif algorithm == 'cpm':
        trainComm = CPM(graph, k)
    else:
        print("invalid algorithm")
        exit(-1)
    metrics["execTime"] = time.time() - start

    trainComm = {int(k): [int(i) for i in v] for k, v in trainComm.items()}
    print(trainComm)
    # evaluation metrics

    metric_time = time.time()
    metrics["f1score"] = Util.f1score(realComm, trainComm)
    print('f1 score time: {}'.format(time.time() - metric_time))
    # metrics["omgIdx"] = Util.omegaIndex(realComm, trainComm)
    metrics["accuracy"] = Util.accuracy(realComm, trainComm)

    print(metrics)
    # output figure and metrics
    Util.outputCommunity(metrics, trainComm, output)
