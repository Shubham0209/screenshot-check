import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import urllib.request

def similarity(url):
    # Open and Convert the input image from BGR to GRAYSCALE

    resp = urllib.request.urlopen(url[0])
    image = np.array(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image,-1)
    image1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('URL Image.jpg', image1)


    # Open and Convert the training-set image from BGR to GRAYSCALE
    resp = urllib.request.urlopen(url[1])
    template = np.array(bytearray(resp.read()), dtype="uint8")
    template = cv.imdecode(template, -1)
    image2 = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    def imageResize(image):
        maxD = 1024
        height,width = image.shape
        aspectRatio = width/height
        if aspectRatio < 1:
            newSize = (int(maxD*aspectRatio),maxD)
        else:
            newSize = (maxD,int(maxD/aspectRatio))
        image = cv.resize(image,newSize)
        return image


    # image1=imageResize(image1)
    # image2=imageResize(image2)
    detector = cv.xfeatures2d.SIFT_create()
    descriptor= cv.xfeatures2d.BEBLID_create(0.75)


    # Find the keypoints
    keypoints1 = detector.detect(image1, None)
    # Compute the descriptors
    keypoints1, descriptors1 = descriptor.compute(image1, keypoints1)


    # Find the keypoints
    keypoints2 = detector.detect(image2, None)
    # Compute the descriptors
    keypoints2, descriptors2 = descriptor.compute(image2, keypoints2)



    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        trees = 5)

    search_params = dict(checks = 50)



    # Create FLANN object
    FLANN = cv.FlannBasedMatcher(indexParams = index_params,
                                    searchParams = search_params)

    # Matching descriptor vectors using FLANN Matcher
    matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                                trainDescriptors = descriptors2,
                                k = 2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = FLANN.knnMatch(queryDescriptors = descriptors2,
                                trainDescriptors = descriptors1,
                                k = 2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)


    def prints(keypoints,detector,descriptor,
            descriptors):
        # Print detector
        print('Detector selected:', detector, '\n')

        # Print descriptor
        print('Descriptor selected:', descriptor, '\n')

        # Print number of keypoints detected
        print('Number of keypoints Detected:', len(keypoints), '\n')

        # Print the descriptor size in bytes
        print('Size of Descriptor:', descriptor.descriptorSize(), '\n')

        # Print the descriptor type
        print('Type of Descriptor:', descriptor.descriptorType(), '\n')

        # Print the default norm type
        print('Default Norm Type:', descriptor.defaultNorm(), '\n')

        # Print shape of descriptor
        print('Shape of Descriptor:', descriptors.shape, '\n')

    print('\nInput image:\n')
    prints(keypoints = keypoints1,detector=detector,descriptor=descriptor,descriptors = descriptors1)

    print('\nTemplate image:\n')
    prints(keypoints = keypoints2,detector=detector,descriptor=descriptor,descriptors = descriptors2)

    score=(len(topResults) /min(len(keypoints1),len(keypoints2))) * 100
    print("GOOD Matches:", len(topResults))
    print("How good it's the match: ", score)


    # Draw only "good" matches
    output = cv.drawMatchesKnn(img1 = image1,
                                    keypoints1 = keypoints1,
                                    img2 = image2,
                                    keypoints2 = keypoints2,
                                    matches1to2 = topResults,
                                    outImg = None,
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchesThickness=2)




    """ if (descriptor == 'SIFT') or (descriptor == 'SURF') or (descriptor == 'KAZE'):
        normType = cv.NORM_L2
    else:
        normType = cv.NORM_HAMMING

    # Create BFMatcher object
    BFMatcher = cv.BFMatcher(normType = normType,
                                crossCheck = True)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.match(queryDescriptors = descriptors1,
                                trainDescriptors = descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)

    # Draw first 30 matches
    output = cv.drawMatches(img1 = image1,
                                    keypoints1 = keypoints1,
                                    img2 = image2,
                                    keypoints2 = keypoints2,
                                    matches1to2 = matches[:30],
                                    outImg = None,
                                    flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    """

    # Create a new figure
    plt.figure()
    plt.axis('off')
    plt.imshow(output)

    plt.imsave(fname = 'Figures/%s-with-%s.png' % ('FLANN', descriptor),
            arr = output,
            dpi = 300)

    # Close it
    plt.close()

    return score

out=similarity(['https://i.imgur.com/E3KCWtp.png', 'https://i.imgur.com/uOYjnKo.jpg'])
print(out)