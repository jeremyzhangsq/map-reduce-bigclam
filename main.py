import sys
import time
import Util
from CPM import CPM
from LC import LC
from BigClam import bigClam
from NMF import NMF


if __name__ == '__main__':

    ufile = sys.argv[1] # network filename
    cfile = sys.argv[2] # ground truth community filename
    algorithm = sys.argv[3] # select an algorithm. option: bigclam, nmf, lc, cpm
    k = int(sys.argv[4]) # number of community to detect
    output = sys.argv[5] # output filename

    matrics = dict()

    start = time.time()
    graph = Util.readNetwork(ufile) # return a n*n matrix
    realComm = Util.readCommunity(cfile) # return a n*k matrix
    matrics["readTime"] = time.time()-start

    start = time.time()
    if algorithm == 'bigclam':
        trainComm = bigClam(graph, k)
    elif algorithm == 'nmf':
        trainComm = NMF(graph, k)
    elif algorithm == 'lc:':
        trainComm = LC(graph, k)
    elif algorithm == 'cpm':
        trainComm = CPM(graph, k)
    else:
        print("invalid algorithm")
        exit(-1)
    matrics["execTime"] = time.time()-start

    # evaluation matrics
    matrics["f1score"] = Util.f1score(realComm,trainComm)
    matrics["omgIdx"] = Util.omegaIndex(realComm,trainComm)
    matrics["accuracy"] = Util.accuracy(realComm,trainComm)

    Util.writeMetrics(output, matrics)
    Util.visualize(trainComm)