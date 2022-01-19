# import the necessary packages
import numpy as np
import urllib.request
import cv2
 

def return_positions_confident(urls):
    
    
    resp = urllib.request.urlopen(urls[0])
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image=image.astype(np.uint8)
    template = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]


    resp = urllib.request.urlopen(urls[1])
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image=image.astype(np.uint8)
    img_gray = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    

    for k in range(10):

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = ((10-k)/10)
        loc = np.where( res >= threshold)
        color=threshold
        print(len(loc[0]))
        if len(loc[0])>0:
            
            positions=[]
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0,204, 153), 3)
                positions.append(pt)

            cv2.imwrite('detected_screen__'+str(threshold)+'_res_thrashold_minimum.png',img_gray)
            break
            
    #return [positions,threshold,'1.1']
    return threshold*100

#print(return_positions_confident('https://i.imgur.com/JPzWnW7.png','https://i.imgur.com/0n38XpR.png'))