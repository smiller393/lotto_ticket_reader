from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output
import pytesseract
import cv2

def detect_text(image):
    results = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 11 --oem 3')
    # print(results)
    resultNumbers = []
    for i in range(0, len(results['text'])):
    
        x = results['left'][i]
        y = results['top'][i]
        
        w = results['width'][i]
        h = results['height'][i]
        text = results['text'][i]
        line_num = results['line_num']
        # print(text)

        conf = int(float(results['conf'][i]))
        if conf > 1:
            newText = ''
            for c in text:
                if ord(c) < 128 and c.isnumeric():
                    newText = newText + c
                elif c == 'o' or c == 'O':
                    newText = newText + '0'
                elif c == 's' or c == 'S':
                    newText = newText + '5'

                
            if newText != '':
                if len(newText) == 2:
                    resultNumbers.append(int(newText))
                if len(newText) == 4:
                    resultNumbers.append(int(newText[:2]))
                    resultNumbers.append(int(newText[2:]))

                newText = newText + "  Conf=" + str(conf)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, newText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    print(resultNumbers)
    cv2.imshow('result_image', image)

    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 



def improve_ticket_image(imgpath):

    image = cv2.imread(cv2.samples.findFile(imgpath))
    if image is None:
        print('Could not open or find the image: ', imgpath)
        exit(0)
        
    new_image = np.zeros(image.shape, image.dtype)

    # alpha = 1.0
    # beta = 5

    # new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # th = 150 # defines the value below which a pixel is considered "black"
    # non_black_pixels = np.where(
    #     (image[:, :, 0] > th) &
    #     (image[:, :, 1] > th) &
    #     (image[:, :, 2] > th)
    # )

    # # set those pixels to white
    # image[non_black_pixels] = [255, 255, 255]

         # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of black color in HSV
    lower_val = np.array([0,0,0])
    upper_val = np.array([179,255,127])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # invert mask to get black symbols on white background
    mask_inv = cv2.bitwise_not(mask)


    return mask_inv
    # # cv2.imshow('Original Image', mask)
    # cv2.imshow('New Image', mask_inv)
    # # Wait until user press some key
    # cv2.waitKey()

############################################################################################



filename = 'images/ticket1.jpg'

improved_image = improve_ticket_image(filename)
detect_text(improved_image)



