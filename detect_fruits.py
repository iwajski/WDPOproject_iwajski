import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict


def circle_contour(image, contour, color_in):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, color=color_in, thickness=2, lineType=1)
    return image_with_ellipse

# best/ref MARPE score: 4.268
# reference values gives the best/ref score. For final adjustments purposes


def detect_fruits(img_path: str) -> Dict[str, int]:
    banana = 0
    orange = 0
    circles_count = 0
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    if height > width:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, [400, 300])
    img_copy = img  # copy for drawing markers, etc.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gs, cv2.HOUGH_GRADIENT, 1, minDist=70, param1=200, param2=11, minRadius=15,
                               maxRadius=40)  # ref:minDist = 70, param1 = 70, param2 = 11, min = 15, max = 40

    # BANANAS:
    lower_yellow = np.array([19, 100, 100])  # ref:[19, 100, 100]
    upper_yellow = np.array([80, 255, 255])  # ref:[80, 255, 255]

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    biggest_area = 1
    for contour in contours:
        if cv2.contourArea(contour) > biggest_area:
            biggest_area = cv2.contourArea(contour)
    for contour in contours:
        cv2.drawContours(img_copy, contour, -1, (255, 0, 255), 3)
        if cv2.contourArea(contour) > 0.40 * biggest_area:  # ref:40% of biggest area
            banana = banana + 1
            img_copy = circle_contour(img_copy, contour, (255, 0, 0))

    mask_no_yellow = cv2.bitwise_not(mask_yellow)
    # mask_no_yellow = cv2.dilate(mask_no_yellow, kernel=kernel)
    no_yellow = cv2.bitwise_and(img, img, mask=mask_no_yellow)
    indices = np.where(mask_no_yellow == 0)
    no_yellow[indices[0], indices[1], :] = [0, 0, 0]
    no_yellow = cv2.cvtColor(no_yellow, cv2.COLOR_BGR2HSV)

    # ORANGE
    min_color = np.array([7, 175, 175])  # ref:[7, 175, 175]
    max_color = np.array([17, 255, 255])  # ref: [17, 255, 255]
    mask_orange = cv2.inRange(no_yellow, min_color, max_color)
    mask_closed = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 800:  # ref:800
            orange = orange + 1
            img_copy = circle_contour(img_copy, contour, (0, 0, 255))

    circles = np.uint16(np.around(circles))
    min_color = np.array([0, 100, 20])  # ref: 0, 100, 20

    for i in circles[0, :]:
        circle_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.circle(circle_img, (i[0], i[1]), i[2], (255, 255, 255), -1)
        circle_mean = cv2.mean(no_yellow, mask=circle_img)[::-1]
        # print(circle_mean)
        if min_color[1] < circle_mean[2]:
            circles_count = circles_count + 1
            cv2.circle(img_copy, (i[0], i[1]), i[2], (255, 255, 10), 2)
        # circle_color = cv2.bitwise_and(img_hsv, img_hsv, mask=circle_img)
    for contour in contours:
        cv2.drawContours(img_copy, contour, -1, (0, 255, 255), 3)
    apple = circles_count - orange

    # debug messages and visualization of detected fruits

    cv2.namedWindow('Window')
    print("banana: ", banana, "apple: ", apple, " oranges: ", orange, " circles: ", circles_count)
    cv2.imshow('Window', img_copy)
    key_code = cv2.waitKey(100)
    input("Press Enter to continue...")


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
