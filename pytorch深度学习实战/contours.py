import cv2
img = cv2.imread('images\eye.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imshow("binary", binary)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("binary2", binary)
#返回两个值：contours：hierarchy。
"""参数
第一个参数是寻找轮廓的图像；
第二个参数表示轮廓的检索模式
cv2.RETR_EXTERNAL 表示只检测外轮廓。
cv2.RETR_LIST 检测的轮廓不建立等级关系。
cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面
的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的
边界也在顶层。
cv2.RETR_TREE 建立一个等级树结构的轮廓。
第三个参数 method 为轮廓的近似办法。
cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素
位置差不超过 1，即 max（abs（x1-x2），abs（y2-y1））==1。
cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向
的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需 4 个点来保
存轮廓信息。
cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS
使用 teh-Chinl chain 近似算法。
返回值
cv2.findContours() 函数返回两个值，一个是轮廓本身，还有一个是每条
轮廓对应的属性。
"""




