import numpy as np

def bigClam(graph, k):
    epsilon = 10 ** (-8)  # background edge propability in sec. 4
    delta = np.sqrt(-np.log(1 - epsilon))  # threshold to determine user-community edge
    pass