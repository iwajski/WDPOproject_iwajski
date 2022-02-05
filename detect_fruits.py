import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict



def detect_fruits(img_path: str) -> Dict[str, int]:
    apple = 0
    banana = 0
    orange = 0
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    if height > width:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, [400, 300])
    cv2.namedWindow('window_canny')
    cv2.namedWindow('window_th')
    cv2.namedWindow('window_og')
    kernel = np.ones((1, 1), np.uint8)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cl = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    th3 = cv2.adaptiveThreshold(img_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 7)
    th3 = cv2.medianBlur(img_gs, 7)
    canny = cv2.Canny(th3, 100, 200)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, minDist=75, param1=200,  param2=11,  minRadius=10, maxRadius = 40)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(th3, (i[0], i[1]), i[2], (255, 255, 10), 2)
        # draw the center of the circle
        cv2.circle(canny, (i[0], i[1]), 2, (0, 0, 255), 3)
        if 60 > img_cl[i[1], i[0], 1] > 10:
            orange = orange+1

    lower_orange = np.array([10, 20, 20])
    upper_orange = np.array([60, 255, 255])
    mask_orange = cv2.inRange(img_cl, lower_orange, upper_orange)
    result = cv2.bitwise_and(img_cl, img_cl, mask=mask_orange)

    cv2.imshow('window_canny', canny)
    cv2.imshow('window_th', th3)
    cv2.imshow('window_og', result)

    #TODO: Implement detection method.
    key_code = cv2.waitKey(100)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits
    
    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()



