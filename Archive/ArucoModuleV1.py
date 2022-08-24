from inspect import Parameter
import os, numpy as np, cv2, cv2.aruco as aruco, os, shutil

def generateArucoMarkers(amtMarkersToMake=10, markerSize=200, markerDimen=6, totalMarkers=250):
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
    print('Total Number of Aruco Markers Detected: ', numOfMarkers)
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
    pts2 = np.float32([0, 0], [w, 0], [w, h], [0, h])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    
    if drawId:
        cv2.putText(imgOut, str(Id), tl, cv2.FRONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut

def main():
    numOfArucoMarkers = input("How Many Aruco Markers Do You Want To Generate? => ")

    if len(os.listdir('Markers')):
        shutil.rmtree('Markers')
    augDics = loadAugImages("Markers")
    
    os.makedirs('Markers', exist_ok=True)

    if int(numOfArucoMarkers) != 0:
        generateArucoMarkers(int(numOfArucoMarkers))
    else:
        generateArucoMarkers()

    cap = cv2.VideoCapture(-1)
    # imgAug = cv2.imread("Markers/0.jpg")

    while cap.isOpened():
        ret, frame = cap.read()
        arucosFound = findArucoMarkers(frame)
        
        # Loop through all the markers and augment each one
        if len(arucosFound[0]) != 0:
            for bbox, Id in zip(arucosFound[0], arucosFound[1]):
                if (augDics.get(int(Id)) != None):
                    resultingFrame = augmentAruco(bbox, Id, frame, augDics[int(Id)])
                    # print(Id, bbox)
        
        cv2.imshow('Live Aruco Marker Detection', resultingFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()