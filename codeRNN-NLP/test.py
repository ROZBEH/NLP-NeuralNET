import numpy as np
from datetime import datetime
def timenow(A, B):
    TT = str(A).split(' ')[1].split(':')
    TT = [int(TT[0]), int(TT[1]), float(TT[2])]
    QQ = str(B).split(' ')[1].split(':')
    QQ = [int(QQ[0]), int(QQ[1]), float(QQ[2])]
    dif = (QQ[0]-TT[0])*3600 + (QQ[1]-TT[1])*60 + (QQ[2]-TT[2])
    return dif




X1 = np.zeros([300,10000])
A = datetime.now()
Y1 = X1[:,500:]
B = datetime.now()
print timenow(A,B)


X2 = np.zeros(300)
Y2 = np.zeros(300)
A = datetime.now()
Z2 = X2 * Y2
B = datetime.now()
print timenow(A,B)
print X2


