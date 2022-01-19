#python match.py --template logo.png --images image
'''
1. Scale down the source image to make the input image smaller and smaller using “loop”.
2. Secondly using OpenCV for edge detection
3. Thirdly apply template matching and keep track of the match with the highest correlation.
4. At last after looping over all scales get the region with the highest correlation.
'''

# import the necessary packages
import numpy as np
import imutils
import cv2
import urllib.request
import os
import requests



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def caclulate_similarity(urls):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    responses = requests.get(urls[0])
    resp = urllib.request.urlopen(responses.url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image=image.astype(np.uint8)
    image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('Image done')
    print('Shape of image is', image.shape)

    found = None


    # load the template, convert it to grayscale, and detect edges since applying template matching using edges rather than the raw image gives us a substantial boost in accuracy for template matching.
    responses = requests.get(urls[1])
    resp = urllib.request.urlopen(responses.url)
    template = np.asarray(bytearray(resp.read()), dtype="uint8")
    template=template.astype(np.uint8)
    template = cv2.imdecode(template, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    print('Shape of template before resizing', template.shape)
    if template.shape[1] > gray.shape[1]: #check width
        template_resized = image_resize(template, width = gray.shape[1])

    elif template.shape[0] >gray.shape[0]: #check height
        template_resized = image_resize(template, height = gray.shape[0])

    else:
        template_resized =image_resize(template)
    print('Shape of template after resizing', template_resized.shape)
    templateGray = cv2.Canny(template_resized, 50, 200)
    (tH, tW) = templateGray.shape[:2]
    print('Template Done')



    # loop over the scales of the image. we’ll start from 100% of the original size of the image and work our way down to 20% of the original size in 20 equally sized percent chunks.
    for scale in np.linspace(1.0, 3.0, 20)[::-1]:
        # resize the image according to the scale, and keep track of the ratio of the resizing
        print('resizing the original image to', scale)
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            print('image is smaller than the template')
            break


        # detect edges in the resized, grayscale image and apply template matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, templateGray, cv2.TM_CCOEFF)
        # we are interested in just the (x, y)-coordinate of the minimum correlation value, and the (x, y)-coordinate of the maximum correlation value, 
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # if we have found a new maximum correlation value, then update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found

    # Threshold setting, this 11195548 value is tested by some random images
    threshold = 111955484
    if maxVal > threshold:
        #print("match found")
        #multiply the coordinates of the bounding box by the ratio r to ensure that the coordinates match the original dimensions of the input image.
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        path_save = os.path.join('static')
        cv2.imwrite(os.path.join(path_save,'Final.jpg'), image)
        cv2.waitKey(0)

        return 'Match found'
    else:
        #print("no match found")
        return 'No match found'

