import sys


name = sys.argv[1]
outname = sys.argv[2]

file = open(name,"r")
outfile = open(outname,"w")

for line in file.readlines():
	nodes = line.rstrip("\n").split("\t")
	s = ""
	for each in nodes[:-1]:
		s += str(int(each)-1)+"\t"
		print(each)
	s += str(int(nodes[-1])-1)+"\n"
	print(s)
	outfile.write("{}".format(s))


file.close()
outfile.close()