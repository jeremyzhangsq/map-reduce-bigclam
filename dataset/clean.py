import sys


name = sys.argv[1]
cname = sys.argv[2]
outname = sys.argv[3]
coutname = sys.argv[4]

file = open(name,"r")
file.readline()
file.readline()
string = file.readline().rstrip("\n").split(" ")
shape = [int(s) for s in string if s.isdigit()]
n=shape[0]
m=shape[1]
file.readline()
graph = {}
maps = {}
cnt = 0
for i in range(int(m)):
	line = file.readline()
	u,v = line.rstrip("\n").split("\t")
	u = int(u)
	v = int(v)
	if u in graph:
		graph[u].append(v)
	else:
		graph[u] = [v]

	if u not in maps:
		maps[u] = cnt
		cnt += 1
	if v not in maps:
		maps[v] = cnt
		cnt += 1

file.close()
outfile = open(outname,"w")	
outfile.write("{}\t{}\n".format(n,m))
for each in graph:
	for value in graph[each]:
		outfile.write("{}\t{}\n".format(maps[each],maps[value]))

outfile.close()

file = open(cname,"r")
outfile = open(coutname,"w")
for line in file.readlines():
	item = line.rstrip("\n").split("\t")
	s = ""
	for each in item[:-1]:
		each = int(each)
		s += str(int(maps[each]))+"\t"
	s += str(int(maps[int(item[-1])]))+"\n"
	outfile.write(s)
file.close()
outfile.close()