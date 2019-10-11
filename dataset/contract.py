"""
Given a size of comminities k,
shrink the input graph to a demo graph where only k communities exists
"""
import sys

fname = sys.argv[1]
cname = sys.argv[2]
k = int(sys.argv[3])

community = set()
graph = dict()
com = open(cname,"r")
ocom = open("{}{}.txt".format(cname.rstrip(".txt"),k),"w")
for i in range(k):
    line = com.readline()
    ocom.write(line)
    l = line.rstrip("\n").split("\t")
    community = community.union(set(l))
com.close()
ocom.close()

file = open(fname,"r")
n,m = file.readline().rstrip("\n").split("\t")
n = int(n)
m = int(m)
cnt = 0
for i in range(m):
    line = file.readline()
    u, v = line.rstrip("\n").split("\t")
    if u in community and v in community:
        if u in graph:
            graph[u].append(v)
        else:
            graph[u] = [v]
        if v in graph:
            graph[v].append(u)
        else:
            graph[v] = [u]
        cnt += 2
file.close()

ofile = open("{}{}.txt".format(fname.rstrip(".txt"),k),"w")
vs = len(graph)
ofile.write("{}\t{}\n".format(vs,cnt))
for key in graph:
    for value in graph[key]:
        ofile.write("{}\t{}\n".format(key,value))
ofile.close()

