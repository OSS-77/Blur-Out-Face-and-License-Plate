import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr

def detect_LP(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    필터링, edge detection
    '''
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

    '''
    Contours 찾기
    '''
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    location = None
    for contour in contours:
      approx = cv2.approxPolyDP(contour, 10, True)
      if len(approx) == 4:
        location = approx
        break

    '''
    빈 마스크 생성 후 contours 그리기
    '''
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask = mask)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    # 번호판으로 인식된 부분만 크롭
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+3, y1:y2+3]

    return cropped_image

def ocr_image(img):
    reader = easyocr.Reader(['ko','en'])
    result = reader.readtext(img)

    return result[0][1]

'''
입력한 경로의 이미지로부터 license plate 검출해 크롭
'''
license_plate = detect_LP('data/car_test.jpeg')
# plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
# plt.show()

'''
위의 검출된 번호판 이미지로부터 OCR
'''
license_number = ocr_image(license_plate)
print(license_number)