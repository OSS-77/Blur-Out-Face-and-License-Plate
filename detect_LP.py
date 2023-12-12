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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

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
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # 번호판으로 인식된 부분만 크롭
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 3, y1:y2 + 3]

    return img, location

def ocr_image(img):
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(img)

    return result[0][1]

def mosaic(src, ratio):
    """
    ### 모자이크 기능
    :param src: 이미지 소스
    :param ratio: 모자이크 비율
    :return: 모자이크가 처리된 이미지
    """
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return mosaic_img

def mosaic_area(src, x, y, width, height, ratio):
    """
    ### 부분 모자이크 기능
    :param src: 이미지 소스
    :param x: 가로축 모자이크 시작 범위
    :param y: 세로축 모자이크 시작 범위
    :param width: 모자이크 범위 넓이
    :param height: 모자이크 범위 폭
    :param ratio: 모자이크 비율
    :return: 부분 모자이크가 처리된 이미지
    """
    mosaic_area_img = src.copy()
    mosaic_area_img[y:y + height, x:x + width] = mosaic(mosaic_area_img[y:y + height, x:x + width], ratio)
    return mosaic_area_img

def opencv_img_save(img, save_img_path, save_img_name):
    """
    ### 처리 이미지 저장 기능
    :param img: 저장할 이미지
    :param save_img_path: 이미지 저장 경로
    :param save_img_name: 저장할 이미지 명
    """
    cv2.imwrite(save_img_path + save_img_name, img)

def process_image(input_image_path):
    img, location = detect_LP(input_image_path)

    # 모자이크 비율
    mosaic_ratio = 0.05

    # 원본 이미지 로드
    original_image = img.copy()

    # 번호판 부분을 검출하고 모자이크 처리
    if location is not None:
        (x, y, w, h) = cv2.boundingRect(location)
        original_image = mosaic_area(original_image, x, y, w, h, mosaic_ratio)

    # 결과 이미지 저장
    output_image_path = 'data/mosaic_result.jpg'
    opencv_img_save(original_image, output_image_path, 'mosaic_result.jpg')

    # 결과 이미지 반환
    return output_image_path

if __name__ == "__main__":
    input_image_path = 'data/car_test.jpeg'
    result_image_path = process_image(input_image_path)
    print(f"결과 이미지 저장 완료: {result_image_path}")
