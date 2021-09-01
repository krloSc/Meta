import time
import numpy as np

prev_time = time.time()
x=np.random.rand(50,2)
values = np.random.rand(31000000,2)

for i in values:
    delta_x = x[:,0] - i[0]
    delta_y = x[:,1] - i[1]
    y = np.sqrt(delta_x**2 + delta_y**2)

z=np.min(y)

print(y)
print(z)
print(time.time()-prev_time)
