from inspect import Parameter
import os, numpy as np, cv2, cv2.aruco as aruco, os, shutil

def generateArucoMarkers(amtMarkersToMake=10, markerSize=350, markerDimen=6, totalMarkers=250):
    markerAttributes = getattr(aruco, f'DICT_{markerDimen}X{markerDimen}_{totalMarkers}')
    # marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    markerDict = aruco.Dictionary_get(markerAttributes)
    
    for Id in range(amtMarkersToMake):
        markerImage = aruco.drawMarker(markerDict, Id, markerSize)
        # cv2.imshow("img", marker_image)
        cv2.imwrite(f"Markers/{Id}.png", markerImage)
        # cv2.waitKey(0)

def loadAugImages(path):
    myList = os.listdir(path)
    numOfMarkers = len(myList)
    print('Total Number of Augmented Images Loaded: ', numOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def findArucoMarkers(img, markerDimen=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerDimen}X{markerDimen}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParams = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParams)

    # print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]        

def augmentAruco(bbox, Id, img, imgAug, drawId=True):
    """Image Augment Reality/Homography (https://www.youtube.com/watch?v=v5a7pKSOJd8)"""
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    
    if drawId:
        # print(Id, tl)
        cv2.putText(imgOut, str(Id), list(map(int, tl)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut

def midpointFromDouble(pnt):
    return list(map(int, [pnt[0] / 2, pnt[1] / 2]))

def main():
    numOfArucoMarkers = input("How Many Aruco Markers Do You Want To Generate? => ")

    if ('Markers' in os.listdir('./')):
        shutil.rmtree('Markers')
    
    os.makedirs(os.path.join('Markers'), exist_ok=True)
    
    augDics = loadAugImages("AugImages")

    if int(numOfArucoMarkers) != 0:
        generateArucoMarkers(int(numOfArucoMarkers))
    else:
        generateArucoMarkers()

    # Regular Video Capture    
    # cap = cv2.VideoCapture(-1) // Could be this too!
    # cap = cv2.VideoCapture(0)

    # Video Capture With Set Resolution, FPS, and Camera Settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Resolution Settings:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Camera Settings:
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    # cap.set(cv2.CAP_PROP_CONTRAST, 150)
    # cap.set(cv2.CAP_PROP_SATURATION, 150)
    # cap.set(cv2.CAP_PROP_HUE, 150)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 150)
    # cap.set(cv2.CAP_PROP_GAIN, 150)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE, 150)
    # cap.set(cv2.CAP_PROP_BACKLIGHT, 150)
    # cap.set(cv2.CAP_PROP_ZOOM, 150)
    # cap.set(cv2.CAP_PROP_FOCUS, 150)
    
    while cap.isOpened():
        ret, frame = cap.read()
        arucosFound = findArucoMarkers(frame)
        
        # print(arucosFound[0])
        
        # Loop through all the markers and augment each one
        if len(arucosFound[0]) != 0:
            for bbox, Id in zip(arucosFound[0], arucosFound[1]):
                if (int(Id) in augDics.keys()):
                    frame = augmentAruco(bbox, Id, frame, augDics[int(Id)])
                    
                    # print(Id, bbox)
                    
                    ### Draw Bounding Box and Centroid on the Aruco Marker
                    # Bounding Box Coordinates
                    # tl = bbox[0][0][0], bbox[0][0][1]
                    # tr = bbox[0][1][0], bbox[0][1][1]
                    # br = bbox[0][2][0], bbox[0][2][1]
                    # bl = bbox[0][3][0], bbox[0][3][1]
                    
                    # cx = int((int(tr[0]) + int(tl[0])) / 2)
                    # cy = int((int(tr[1]) + int(br[1])) / 2)
                    
                    # Centroid Point of The Bounding Box
                    # cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            # print(bbox[0][1][0], bbox[0][1][1])
        
        cv2.imshow('Live Aruco Marker Detection', frame)
        # cv2.imwrite(f"lo.png", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# Expand the contents of the view;
# DICT_4X4_50 
# Python: cv.aruco.DICT_4X4_50
# DICT_4X4_100 
# Python: cv.aruco.DICT_4X4_100
# DICT_4X4_250 
# Python: cv.aruco.DICT_4X4_250
# DICT_4X4_1000 
# Python: cv.aruco.DICT_4X4_1000
# DICT_5X5_50 
# Python: cv.aruco.DICT_5X5_50
# DICT_5X5_100 
# Python: cv.aruco.DICT_5X5_100
# DICT_5X5_250 
# Python: cv.aruco.DICT_5X5_250
# DICT_5X5_1000 
# Python: cv.aruco.DICT_5X5_1000
# DICT_6X6_50 
# Python: cv.aruco.DICT_6X6_50
# DICT_6X6_100 
# Python: cv.aruco.DICT_6X6_100
# DICT_6X6_250 
# Python: cv.aruco.DICT_6X6_250
# DICT_6X6_1000 
# Python: cv.aruco.DICT_6X6_1000
# DICT_7X7_50 
# Python: cv.aruco.DICT_7X7_50
# DICT_7X7_100 
# Python: cv.aruco.DICT_7X7_100
# DICT_7X7_250 
# Python: cv.aruco.DICT_7X7_250
# DICT_7X7_1000 
# Python: cv.aruco.DICT_7X7_1000
# DICT_ARUCO_ORIGINAL 
# Python: cv.aruco.DICT_ARUCO_ORIGINAL
# DICT_APRILTAG_16h5 
# Python: cv.aruco.DICT_APRILTAG_16h5
# 4x4 bits, minimum hamming distance between any two codes = 5, 30 codes