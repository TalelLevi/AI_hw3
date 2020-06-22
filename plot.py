from matplotlib import pyplot as plt

prediction = (1, 1)

plt.scatter(1, 1, c='r', label='x to predict')
plt.scatter(6, 1, c='y', label='only in DS2')
plt.scatter(2, 1, c='b', label='only in DS1')
plt.scatter((1, 1, 2), (5, 2, 2), c='g', label='in both DS1 and DS2')
plt.xlabel('feature a')
plt.ylabel('feature b')
plt.legend()
plt.show()
