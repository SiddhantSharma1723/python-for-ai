import numpy as np
rng = np.random.default_rng()
scores = rng.integers(0,101,1000000)
print(scores)
def analysis_loop(scores):
    l = len(scores)

    total = 0
    for s in scores:
        total += s
    mean = total/l

    var_sum = 0
    for s in scores:
        var_sum += (mean - s)**2
    var = var_sum/l
    std = var**0.5
    z = []
    for s in scores:
        z_score = (s-mean)/std
        z.append(z_score)

    return mean, std, z

def analysis_vectorised(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    z = (scores - mean)/std
    a = np.where(z > 2)
    boolean_shi = scores[a]
    no_shi = len(boolean_shi)
    return mean, std ,z ,no_shi


meanl,stdl,zl = analysis_loop(scores)
meanv,stdv,zv, _ = analysis_vectorised(scores)

print("Means match:",np.isclose(meanl,meanv))
print("Std match:",np.isclose(stdl,stdv))
print("Z-scores match:",np.allclose(zl,zv))

import time
start1 = time.time()
analysis_loop(scores)
stop1 = time.time()
loop = stop1-start1
print("Loop time:", stop1-start1,"Seconds")

start = time.time()
analysis_vectorised(scores)
stop = time.time()
vector = stop-start
print("Vectorised time:", stop-start,"Seconds")

if vector<loop:
    print("Vectorised is faster")
else:
    print("Loop is faster")