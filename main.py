import os
from detect_LP import detect_LP, ocr_image
from detect_face import detectFace
from mosaic import mosaic_area, opencv_img_save
import cv2

def process_image(input_image_path):
    img = cv2.imread(input_image_path)

    # 얼굴과 번호판을 찾기
    try:
        faces = detectFace(img)
    except:
        print('Face Not Detected')
        faces=[]
    try:
        license_plate_area, location = detect_LP(img)
    except:
        print('License Plate Not Detected')
        location=None

    # 모자이크 비율
    mosaic_ratio = 0.05

    # 원본 이미지 로드
    original_image = img

    # 얼굴이 있는 경우 얼굴 부분 모자이크 처리
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            original_image = mosaic_area(original_image, x, y, w, h, mosaic_ratio)

    # 번호판 부분을 검출하고 모자이크 처리
    if location is not None:
        (x, y, w, h) = cv2.boundingRect(location)
        original_image = mosaic_area(original_image, x, y, w, h, mosaic_ratio)
        # 번호판 ocr 결과 출력
        try:
            lp_number = ocr_image(license_plate_area)
            print(f'Detected License Plate: {lp_number}')
        except:
            pass

    # 결과 이미지 저장
    wd=os.getcwd()
    output_image_path = wd+'/data/mosaic_result.jpg'
    opencv_img_save(original_image, output_image_path, '_mosaic.jpg')
    print(f'Output is stored in {output_image_path}_mosaic.jpg')
    # 결과 이미지 반환
    return output_image_path

wd=os.getcwd()
result = process_image(wd+'/data/car_test.jpeg')
