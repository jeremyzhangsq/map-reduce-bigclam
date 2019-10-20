import sys
import time
import Util
from CPM import CPM
from LC import LC
from BigClam import bigClam
from NMF import NMF


if __name__ == '__main__':

    # input params
    ufile = sys.argv[1] # network filename
    cfile = sys.argv[2] # ground truth community filename
    algorithm = sys.argv[3] # select an algorithm. option: bigclam, nmf, lc, cpm
    k = int(sys.argv[4]) # number of community to detect
    output = sys.argv[5] # output filename

    metrics = dict()

    # load two matrix
    start = time.time()
    graph = Util.Graph(ufile,cfile)
    metrics["algorithm"] = algorithm
    metrics["readTime"] = time.time() - start

    # main algorithm
    start = time.time()
    if algorithm == 'bigclam':
        trainComm = bigClam(graph, k)
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
    realComm = graph.community
    trainComm = {int(k):[int(i) for i in v] for k,v in trainComm.items()}
    print(trainComm)
    # evaluation metrics
    metrics["f1score"] = Util.f1score(realComm, trainComm)
    metrics["omgIdx"] = Util.omegaIndex(realComm, trainComm)
    metrics["accuracy"] = Util.accuracy(realComm, trainComm)

    print(metrics)
    # output figure and metrics
    # Util.writeMetrics(output, metrics)
    # Util.outputCommunity(trainComm,output)