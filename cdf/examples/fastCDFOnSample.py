import numpy as np
import cdf

nbSimul = 100000
x = np.random.normal(size=(2, nbSimul))
# y =1 for simple CDF
y = np.ones([nbSimul])

ecdf = cdf.fastCDFOnSample(x, y)
