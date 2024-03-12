 # Importing the Libraries
import cv2
import numpy as np

# Function for geting information of April tag such as ID and orientation
def detect_id(image):

    ret, img_binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    if 255 in image[51:75, 51:75]:
     print("The position of the TAG is 'top_left'")
     orintation = "TL"
     
    # Top Right
    elif 255 in image[126:150, 51:75]:
        print("The position of the TAG is 'top_right")
        orintation = "TR"

    # Bottom right
    elif 255 in image[51:75,126:150 ]:
        print("The position of the TAG is 'bottom_right")
        orintation = "BR"
        
    # Bottom Left
    else:
        print("The position of the TAG is 'bottom_left")
        orintation = "BL"


    
    
    return orintation

# Function to change the rotate the image
def changeOrient(image, orient):
    if orient == 1:
        requried_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orient == 2:
        requried_image = cv2.rotate(image, cv2.ROTATE_180)
    elif orient == 3:
        requried_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        requried_image = image
    return requried_image

# Function to detect the counters and the corners
def detect_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to grayscale
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # Converting to binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Detecting contours

    false_contours = []  # Creating a list to detect false contours

    # Using tree hierarchy to detect contour with no parent and no child
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:  # The last 2 element of the hierarchy matrix
            false_contours.append(i)

    # If the contour is not in false contour, then it is a true contour
    contours = [c for i, c in enumerate(contours) if i not in false_contours]

    # sorting them and accessing the first three contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # The final filter is to converting it to approximate rectangle
    final_contours = []

    # Accessing each contour individually
    for c in sorted_contours:
        epsilon = cv2.arcLength(c, True)  # Finding the epsilon
        approx = cv2.approxPolyDP(c, epsilon * .015, True)  # Multiplying epsilon with calibrated constant
        if len(approx) == 4:  # If the polygon has 4 sides it is a rectangle
            final_contours.append(approx)  

    corners = []
    for shape in final_contours:
        points = []
        for p in shape:
            points.append([p[0][0], p[0][1]])
        corners.append(points)

    return final_contours, corners






# Function to find the homography of fixed square image
def homography(corners, dim=500):
    # Define the eight points to compute the homography matrix
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    # ccw corners
    xp = [0, dim, dim, 0]
    yp = [0, 0, dim, dim]
    #homography with svd
    xp = np.array([0,dim,dim,0])
    yp = np.array([0,0,dim,dim])
    
    A = np.matrix([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                    [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                    [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                    [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                    [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                    [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]], 
                    [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                    [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1,:]/Vh[-1,-1]
    H = np.reshape(l, (3,3))
    return H


# Function to apply warp to change field of view
def warp(H, src, h, w):
    idxy, idxx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([idxx.ravel(), idxy.ravel(), np.ones_like(idxx).ravel()])

    map_ind = H.dot(lin_homg_ind)
    x_map, y_map = map_ind[:-1] / map_ind[-1]
    x_map = x_map.reshape(h, w).astype(np.float32)
    y_map = y_map.reshape(h, w).astype(np.float32)

    x_map[x_map >= src.shape[1]] = -1
    x_map[x_map < 0] = -1
    y_map[y_map >= src.shape[0]] = -1
    x_map[y_map < 0] = -1

    return_img = np.zeros((h, w, 3), dtype="uint8")
    for x_new in range(w):
        for y_new in range(h):
            x = int(x_map[y_new, x_new])
            y = int(y_map[y_new, x_new])

            if x == -1 or y == -1:
                pass
            else:
                return_img[y_new, x_new] = src[y, x]
    return return_img



# Function to impose the image on the AR TAG
def impose(frame, contour, color):
    cv2.drawContours(frame, [contour], -1, (color), thickness=-1)
    return frame

# Reading the image and the video
img_testudo = cv2.imread('/home/enigma/Downloads/CV_1/nisarg15_Project1/testudo.png')
video = cv2.VideoCapture('/home/enigma/Downloads/CV_1/nisarg15_Project1/1tagvideo.mp4')

# Camera parameters 
K = np.array([[1346.101, 0, 0],
              [0, 1355.933, 654.8987],
              [0, 0, 1]])


while video.isOpened():
    ret, frame = video.read()
    result = cv2.VideoWriter('Tag0.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (int(video.get(3)), int(video.get(4))))
    [all_cnts, cnts] = detect_contour(frame) 
    for i, tag in enumerate(cnts):
        H = np.linalg.inv(homography(tag))# Homography      
   
        img_square = warp(H, frame, 500, 500)       # Warping image
        img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
        
        orient = detect_id(img_gray)
        img_rotate = changeOrient(img_testudo, orient) #Check orientation and rotate the image
        
        frame1 = warp(homography(tag, img_rotate.shape[0]), changeOrient(img_testudo, orient), frame.shape[0], frame.shape[1])
        frame2 = impose(frame, all_cnts[i], 1)
        superimposed_frame = cv2.bitwise_or(frame1, frame2)
        cv2.imshow("Testudo", superimposed_frame)
        result.write(frame) 
        if cv2.waitKey(1):
            break
           
        

        

 
        
        

        
                             
                             
    