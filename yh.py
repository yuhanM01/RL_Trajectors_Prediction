import numpy as np

m4_ActionList = np.linspace(0, 358, 180)
aa = np.empty(shape=[0,5], dtype=np.float32)
aa = np.append(aa,[[1,2,3,4,5]], axis=0)
print(aa)
print(m4_ActionList)