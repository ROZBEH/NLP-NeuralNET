inputfile = open("list",'r')
input_ = inputfile.readlines()

freq_voc = []
for item in input_:
	freq_voc.append(item.split('\n')[0])

print freq_voc