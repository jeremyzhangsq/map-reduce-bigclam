import sys
import operator
cname = sys.argv[1]
com = open(cname,"r")
comlist = []
overlap = {}
for line in com.readlines():
    l = line.rstrip("\n").split("\t")
    comlist.append(l)
com.close()

for i in range(len(comlist)):
    for ele in comlist[i]:
        if ele not in overlap:
            overlap[ele] = [i]
        else:
            overlap[ele].append(i)
result = [(k,len(overlap[k])) for k in overlap]
result = sorted(result,key=operator.itemgetter(1),reverse=True)
for k,lens in result[:10]:
    print("{}:{}".format(k,lens))









