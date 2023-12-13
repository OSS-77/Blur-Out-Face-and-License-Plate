import os

import cv2
import numpy as np
import imutils

#detect_face.py
def detectFace(img):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    return faces
#detect_LP.py
def detect_LP(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    license_plate_area = gray[x1:x2 + 3, y1:y2 + 3]

    return license_plate_area, location

def mosaic(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return mosaic_img

def mosaic_area(src, x, y, width, height, ratio):
    mosaic_area_img = src.copy()
    mosaic_area_img[y:y + height, x:x + width] = mosaic(mosaic_area_img[y:y + height, x:x + width], ratio)
    return mosaic_area_img

def opencv_img_save(img, save_img_path, save_img_name):
    cv2.imwrite(save_img_path + save_img_name, img)

def process_image(input_image_path):
    img = cv2.imread(input_image_path)

    # 얼굴과 번호판을 찾기
    faces = detectFace(img)
    license_plate_area, location = detect_LP(img)

    # 모자이크 비율
    mosaic_ratio = 0.05

    # 원본 이미지 로드
    original_image = img.copy()

    # 얼굴이 있는 경우 얼굴 부분 모자이크 처리
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            original_image = mosaic_area(original_image, x, y, w, h, mosaic_ratio)

    # 번호판 부분을 검출하고 모자이크 처리
    if location is not None:
        (x, y, w, h) = cv2.boundingRect(location)
        original_image = mosaic_area(original_image, x, y, w, h, mosaic_ratio)

    # 결과 이미지 저장
    wd=os.getcwd()
    output_image_path = wd+'/data/mosaic_result.jpg'
    opencv_img_save(original_image, output_image_path, 'mosaic_result.jpg')

    # 결과 이미지 반환
    return output_image_path

# if __name__ == "__main__":
#     input_image_path = 'data/car_test.jpeg'
#     result_image_path = process_image(input_image_path)
#     print(f"결과 이미지 저장 완료: {result_image_path}")

