import numpy as np
import cdf

nbSimul = 100000
d = 6
x = np.random.normal(size=(d, nbSimul))
# y =1 for simple CDF
y = np.ones([nbSimul])

ecdf = cdf.fastCDFOnSample(x, y)
