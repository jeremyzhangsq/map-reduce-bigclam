"""
Given a size of comminities k,
shrink the input graph to a demo graph where only k communities exists
"""
import sys
import random

fname = sys.argv[1]
cname = sys.argv[2]
k = int(sys.argv[3])

community = set()
comlist = []
graph = dict()
map = dict()
com = open(cname,"r")
all = []
for line in com.readlines():
    l = line.rstrip("\n").split("\t")
    all.append(l)
com.close()

for i in range(k):
    idx = random.randrange(len(all))
    l = all.pop(idx)
    comlist.append(l)
    community = community.union(set(l))

file = open(fname,"r")
n,m = file.readline().rstrip("\n").split("\t")
n = int(n)
m = int(m)
cnt = 0
id = 0
for i in range(m):
    line = file.readline()
    u, v = line.rstrip("\n").split("\t")
    if u in community and v in community:
        if u in graph:
            graph[u].add(v)
        else:
            graph[u] = set()
            graph[u].add(v)
            map[u] = id
            id += 1
        cnt += 1
file.close()

ocom = open("{}{}.txt".format(cname.rstrip(".txt"),k),"w")
for item in comlist:
    s = ""
    for each in item[:-1]:
        s += str(map[each]) + "\t"
    s += str(map[item[-1]]) + "\n"
    ocom.write(s)
ocom.close()


ofile = open("{}{}.txt".format(fname.rstrip(".txt"),k),"w")
vs = len(graph)
ofile.write("{}\t{}\n".format(vs,cnt))
for key in graph:
    for value in graph[key]:
        ofile.write("{}\t{}\n".format(map[key],map[value]))
ofile.close()

