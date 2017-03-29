import numpy as np
from datetime import datetime
def timenow(A, B):
    TT = str(A).split(' ')[1].split(':')
    TT = [int(TT[0]), int(TT[1]), float(TT[2])]
    QQ = str(B).split(' ')[1].split(':')
    QQ = [int(QQ[0]), int(QQ[1]), float(QQ[2])]
    dif = (QQ[0]-TT[0])*3600 + (QQ[1]-TT[1])*60 + (QQ[2]-TT[2])
    return dif



a = np.ones((400,3124))
b = np.zeros(3124)
b[1] = 1



A = datetime.now()
c = a[:,1]
B = datetime.now()
print timenow(A,B)

A = datetime.now()
d = np.dot(a,b)
B = datetime.now()
print timenow(A,B)

print c.shape
print d.shape