
import cv2
import numpy as np



def tenengrad(img):
    grad_value = 0
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            sx = img[i-1, j-1]*(-1) + img[i-1, j+1] + img[i, j-1]*(-2) + img[i, j+1]*2 + img[i+1, j-1]*(-1) + img[i+1, j+1]  
            sy = img[i-1, j-1] + 2*img[i-1, j] + img[i-1, j+1] - img[i+1, j-1] - 2*img[i+1, j] - img[i+1, j+1]  # y方向梯度
            grad_value += sx * sx + sy * sy
    return grad_value / (cols - 2) / (rows - 2)


def Identification(generated_image):
        if len(generated_image.shape) == 3:
            grayImage = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
        else:
            grayImage = generated_image
        grayImage = cv2.normalize(grayImage, None, 0, 255, cv2.NORM_MINMAX)
        grayImage = np.uint8(grayImage)
        gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
        Canny = cv2.Canny(gaussianBlur, 50, 150)
        dilated = cv2.dilate(Canny, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        Canny = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours, hierarchy = cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(grayImage)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        Object = cv2.bitwise_and(grayImage, grayImage, mask=mask)
        mask_inv = cv2.bitwise_not(mask)
        Artifact = cv2.bitwise_and(grayImage, grayImage, mask=mask_inv)


        return Object, Artifact


def count_zero_pixels(image):
    if len(image.shape) < 3:
        gray = image  
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_nonzero_pixels = np.count_nonzero(gray)
    num_zero_pixels = 64*64 - num_nonzero_pixels
    return num_zero_pixels

def calculate_pixel_sum(image):
    if len(image.shape) < 3:
        gray = image  
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_sum = np.sum(gray)
    return pixel_sum









