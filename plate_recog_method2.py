import cv2
import argparse


class Plate():
    imgPath = ""
    def __init__(self,imgpath):
        self.imgPath = imgpath
        self.rawImage = cv2.imread(imgpath)
        self.possiblePlateArea = []

    def plateDetect(self):
        # Grey
        gray = cv2.cvtColor(self.rawImage,cv2.COLOR_BGR2GRAY)

        # Gaussian smooth image
        gaussian = cv2.GaussianBlur(self.rawImage, (3,3), 0,0,cv2.BORDER_DEFAULT)

        # Median
        median = cv2.medianBlur(gaussian,5)

        # Sobel Edge detect
        # x = cv2.Sobel(median, cv2.CV_32F, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)
        # y = cv2.Sobel(median, cv2.CV_32F, 0, 1, 3, 1, 1, cv2.BORDER_DEFAULT)
        #
        # absX = cv2.convertScaleAbs(x)
        # absY = cv2.convertScaleAbs(y)
        #
        # sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # sobel
        sobel = cv2.Sobel(median, cv2.CV_8U, 1,0, ksize=3)

        ret, binary = cv2.threshold(sobel,170,255, cv2.THRESH_BINARY)

        # 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)
        # 腐蚀一次，去掉细节
        erosion = cv2.erode(dilation, element1, iterations=1)
        # 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations=3)


        cv2.imshow("dilation2",dilation2)


parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="3",help="Source Image file")
img_char, unparsed = parser.parse_known_args()
# print(img_char.img)
plate = Plate(img_char.img)
plate.plateDetect()
cv2.waitKey(10000)

