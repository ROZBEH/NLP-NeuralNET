def bin_range(value, max, min):
	if min <= value < max/float(5):
		return 0
	elif 1*(max/float(5)) <= value < 2*(max/float(5)):
		return 1
	elif 2*(max/float(5)) <= value < 3*(max/float(5)):
		return 2
	elif 3*(max/float(5)) <= value < 4*(max/float(5)):
		return 3
	else:
		return 4