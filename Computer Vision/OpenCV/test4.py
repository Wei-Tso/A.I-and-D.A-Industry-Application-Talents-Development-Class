import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('./image/test10_1.jpg')
rows, cols, ch = img.shape

# 设置标记点和目标点
markpoint = [[1100, 120], [2950, 850], [2750, 3680], [630, 3870]]
dstpoint = [[0, 0], [300, 0], [300, 400], [0, 400]]

# 强调标记点
for i in markpoint:
    cv2.circle(img, tuple(i), 10, (0, 255, 0), -1)

# 转换点的格式
pts1 = np.float32(markpoint)
pts2 = np.float32(dstpoint)

# 生成透视矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)

# 转换
dst = cv2.warpPerspective(img, M, (300, 400))

# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

cv2.imshow("before" , img)
cv2.imshow("after" , dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
