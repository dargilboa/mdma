import numpy as np
import cdf

nbSimul = 100000
x = np.random.normal(size=(2, nbSimul))
# y =1 for simple CDF
y = np.ones([nbSimul])

# define the grid (same in both dimension)
z1 = -2 + 4 * np.arange(20) / 20
z = [z1, z1]
ecdf = cdf.fastCDF(x, z, y)
