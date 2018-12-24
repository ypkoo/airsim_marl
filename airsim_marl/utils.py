import numpy as np

def distance(a, b):
	c = (a[0]-b[0], a[1]-b[1])
	return np.linalg.norm(c)

def airsiwaawdm_setting():
	pass