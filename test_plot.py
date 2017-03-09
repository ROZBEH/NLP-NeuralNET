min_lat = 40.5774
max_lat = 40.9176
min_long = -74.15
max_long = -73.7004

import matplotlib.pyplot as plt

plt.plot([40.75,40.73],[-73.90,-73.80],'ro')
plt.xlim(min_lat, max_lat)
plt.ylim(min_long,max_long)
plt.show()
