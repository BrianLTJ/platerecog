import cv2
import os
import numpy as np

filename = "sua20q03.jpg"


def gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def thresh(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def sobel(img):
    dst = cv2.Sobel(img, 1, 1, 0)
    print(dst)
    return dst


class Plate(object):
    rawImgPath = ""
    rawImg = ""
    sobelImg = None

    rawImgWithSobelContour = None

    plateColor = ""

    # possible plate Area images
    plateArea = []

    # divided char for each possible plate
    ## a empty list [] will be inserted to the list.
    ## format:
    ## [imgChar1,imgChar2...]
    plateChar = []

    def __init__(self, img_path):
        self.rawImgPath = img_path
        self.loadImg()


    def loadImg(self):
        self.rawImg = cv2.imread(self.rawImgPath)


    def detectPlateArea(self):
        # Transform image into gray
        gray = cv2.cvtColor(self.rawImg,cv2.COLOR_BGR2GRAY)

        # Gaussian Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        # Sobel Edge detect
        x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)
        y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, 3, 1, 1, cv2.BORDER_DEFAULT)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        sobeldst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        sobeldst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        ## Threshold sobel image
        self.sobelthresh = thresh(sobeldst)

        self.rawImgWithSobelContour = self.rawImg

        conts = cv2.findContours(self.sobelthresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(conts)

        # filter the contours with width/height in a certain range
        possible_conts = []
        for c in conts:
            ### Read each contour area
            (x,y,w,h) = cv2.boundingRect(c)
            ar = float(w) / float(h)

            ### Width Height w/h ratio
            ### Normal plate size 440*140
            if w>=80 and h>=20 and ar >=2.0 and ar <=4.0:
                possible_conts.append(c)
                #### Crop possible plate area
                plateimg = self.rawImg[y:y + h, x:x + w]
                self.plateArea.append(plateimg)

        # cv2.drawContours(self.sobelthresh, possible_conts, -1, (0, 255, 0), 3)

    def plateAreaTransform(self):
        ## Transform plate Area to align
        i = 0
        for possiblePlate in self.plateArea:

            plate = possiblePlate
            plate = cv2.cvtColor(possiblePlate,cv2.COLOR_BGR2GRAY)
            plate = cv2.GaussianBlur(plate,(5,5),1)
            plate = thresh(plate)
            # cv2.imshow(str(i),plate)
            i = i+1
            cv2.waitKey(1000)
            cnts = cv2.findContours(plate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
            cv2.drawContours(plate,cnts,-1,(0,255,0),3)
            #
            #
            # validCnts = []
            #
            # for c in cnts:
            #     (x,y,w,h) = cv2.boundingRect(c)
            #     ar = float(w)/float(h)
            #
            #     if w >= 140 and h >= 20 and ar >= 2.0 and ar <= 4.0:
            #         validCnts.append(c)
            #
            #
            # drawedImg = possiblePlate
            # cv2.drawContours(drawedImg,validCnts, -1, (0, 255, 0), 3)
            cv2.imshow(str(i),plate)
            # i=i+1

    def splitChar(self):
        i = 0
        for possiblePlate in self.plateArea:
            (ih, iw, ip) = possiblePlate.shape
            plate = possiblePlate
            plate = cv2.cvtColor(possiblePlate, cv2.COLOR_BGR2GRAY)

            plate = cv2.GaussianBlur(plate, (5, 5), 1)
            plate = thresh(plate)

            # cv2.imshow("Before" + str(i), plate)

            thr_list = plate.tolist()
            plate_mat = []
            # print(thr_list)
            for line_item in thr_list:
                line_mat = []
                for item in line_item:

                    if item>= 180:
                        line_mat.append(1)
                    else:
                        line_mat.append(0)

                plate_mat.append(line_mat)

            # Calculate Accum var in row
            accum_line = []
            for line in plate_mat:
                sum = 0
                for i in line:
                    sum = sum + i
                accum_line.append(sum)

            h = len(plate_mat)
            w = len(plate_mat[0])
            accum_col = []
            for col in range(w):
                sum = 0
                for row in range(h):
                    sum = sum + plate_mat[row][col]

                accum_col.append(sum)

            # cv2.imshow("After"+str(i), plate)
            print(accum_line)
            print(accum_col)

            i = i + 1

    def splitCharByContour(self):
        j = 0
        for possiblePlate in self.plateArea:
            (ih, iw, ip) = possiblePlate.shape
            print("Plate shape: width:",iw,"height:",ih)
            plate = possiblePlate
            plate = cv2.cvtColor(possiblePlate, cv2.COLOR_BGR2GRAY)

            plate = cv2.GaussianBlur(plate, (5, 5), 1)
            plate = thresh(plate)

            # Sobel Edge detect
            x = cv2.Sobel(plate, cv2.CV_32F, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)
            y = cv2.Sobel(plate, cv2.CV_32F, 0, 1, 3, 1, 1, cv2.BORDER_DEFAULT)

            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)

            sobeldst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            sobeldst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

            conts = cv2.findContours(thresh(sobeldst), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
            print("Contour area:",len(conts))

            possibleChar = []


            for c in conts:
                (x,y,w,h) = cv2.boundingRect(c)
                try:
                    epsilon = 0.05 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    cv2.drawContours(possiblePlate,approx,1,(0,255,0),2)
                except:
                    pass
                hByW = float(h)/float(w)
                print(x,y,w,h)

                if hByW >=1.0 and hByW <=3.0 and w>0.1*iw and h>0.2*ih:
                    possibleChar.append(plate[y:y+h,x:x+w])
            cv2.imshow("Plate " + str(j), possiblePlate)
            print("Possible char area:",len(possibleChar))
            for i in range(len(possibleChar)):
                cv2.imshow("Char"+str(i),possibleChar[i])
                cv2.waitKey(5000)

            j=j+1


    def getContourImg(self):
        return self.rawImgWithSobelContour


test = Plate(img_path=filename)
test.detectPlateArea()
# test.plateAreaTransform()
# test.splitChar()
test.splitCharByContour()

cv2.waitKey(0)
cv2.destroyAllWindows()







