import os
from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output
import pytesseract
import cv2

def detect_text(image):
    results = pytesseract.image_to_data(image, output_type=Output.DICT, config='')
    resultNumbers = []
    for i in range(0, len(results['text'])):
    
        x = results['left'][i]
        y = results['top'][i]
        
        w = results['width'][i]
        h = results['height'][i]
        text = results['text'][i]
        line_num = results['line_num']

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
                if len(newText) == 6:
                    resultNumbers.append(int(newText[:2]))
                    resultNumbers.append(int(newText[2:4]))
                    resultNumbers.append(int(newText[4:]))

                if len(newText) == 8:
                    resultNumbers.append(int(newText[:2]))
                    resultNumbers.append(int(newText[2:4]))
                    resultNumbers.append(int(newText[4:6]))
                    resultNumbers.append(int(newText[6:]))
                if len(newText) == 10:
                    resultNumbers.append(int(newText[:2]))
                    resultNumbers.append(int(newText[2:4]))
                    resultNumbers.append(int(newText[4:6]))
                    resultNumbers.append(int(newText[6:8]))
                    resultNumbers.append(int(newText[8:]))



                newText = newText + "  Conf=" + str(conf)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, newText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)


    lotto_numbers = resultNumbers[-60:]

    return len(resultNumbers),lotto_numbers,image

def improve_ticket_image(imgpath):

    image = cv2.imread(cv2.samples.findFile(imgpath))
    if image is None:
        print('Could not open or find the image: ', imgpath)
        exit(0)
        
    # image = image.img_to_array(image, dtype='uint8')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)


    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 4)

    # # invert mask to get black symbols on white background
    mask_inv = cv2.bitwise_not(thresh)

    return mask_inv 


############################################################################################

final_result_list = []
# list_of_files = os.listdir('images')
# for filename in list_of_files:
#     f = os.path.join('images', filename)
#     if f[-4:] == '.jpg':
f = 'images/PXL_20220729_232501519.jpg'
for n in range(0,5):
    improved_image = improve_ticket_image(f)
    count,nums,image = detect_text(improved_image)
    if count >= 60:
        final_result_list.append(nums)
        print(count)
        break
    # print(count)
    if n == 4:
        print(count)
        print(f)
    #     cv2.imshow('result_image', image)

    #     # #waits for user to press any key 
    #     # #(this is necessary to avoid Python kernel form crashing)
    #     cv2.waitKey(0) 

    #     # #closing all open windows 
    #     cv2.destroyAllWindows() 



# print(final_result_list)
# print(len(final_result_list))

for ticket in final_result_list:
    gameslist = np.array_split(ticket,10)
    for game in gameslist:
        winningnums = 0
        megaball_match = False
        for x,num in enumerate(game):
            if x == 5:
                if num == 14:
                    megaball_match = True
            else: 
                if num in [13,36,45,57,67]:
                    winningnums += 1
        if winningnums >= 3 or megaball_match: 
            print(winningnums,megaball_match,game)
            
        list = [str(x) for x in game] 
        # print(",".join(list))

