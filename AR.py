import cv2
import numpy as np


# Function for geting information of April tag such as ID and orientation

# Function to detect contours and sorting them
def detect_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(gray, 190, 255, 0)
    all_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # to exclude the wrong contours
    wrong_cnts = []
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            wrong_cnts.append(i)
    cnts = [c for i, c in enumerate(all_cnts) if i not in wrong_cnts]

    # sort the contours to include only the three largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    return_cnts = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * .015, True)
        if len(approx) == 4:
            return_cnts.append(approx)

    corners = []
    for shape in return_cnts:
        points = []
        for p in shape:
            points.append([p[0][0], p[0][1]])
        corners.append(points)

    return return_cnts, corners


def find_homography(p1, p2):
        A = []
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)  # Using SVD file
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)
        return H



def projection_matrix(K, H):
    h1 = H[:, 0]
    h2 = H[:, 1]

    K = np.transpose(K)

    inv_K = np.linalg.inv(K)
    a = np.dot(inv_K, h1)
    c = np.dot(inv_K, h2)
    lamda = 1 / ((np.linalg.norm(a) + np.linalg.norm(c)) / 2)

    B_T = np.dot(inv_K, H)

    if np.linalg.det(B_T) > 0:
        B = 1 * B_T
    else:
        B = -1 * B_T

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    r1 = lamda * b1
    r2 = lamda * b2
    r3 = np.cross(r1, r2)
    t = lamda * b3
    P = np.dot(K, (np.stack((r1, r2, r3, t), axis=1)))

    return P


video = cv2.VideoCapture('C:/Users/nisar/Desktop/Projects/nisarg15_Project1/1tagvideo.mp4')
# Camera intrinsic parameters
K = np.array([[1346.101, 0, 0],
              [0, 1355.933, 654.8987],
              [0, 0, 1]])

while video.isOpened():
        ret, frame = video.read()
        result = cv2.VideoWriter('Cube.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (int(video.get(3)), int(video.get(4))))
        [all_cnts, cnts] = detect_contour(frame)  # Detect Contours
        dst_pts = [[0, 0], [0, 500], [500, 500], [500, 0]]  # Feeding the destination size (Image size)
        H_cube = find_homography(dst_pts, cnts[0])
        P = projection_matrix(K,H_cube)
        
        # Getting points in 3D from projection matrix
        x1, y1, z1 = np.matmul(P, [0, 0, 0, 1])
        x2, y2, z2 = np.matmul(P, [0, 500, 0, 1])
        x3, y3, z3 = np.matmul(P, [500, 0, 0, 1])
        x4, y4, z4 = np.matmul(P, [500, 500, 0, 1])
        x5, y5, z5 = np.matmul(P, [0, 0, -500, 1])
        x6, y6, z6 = np.matmul(P, [0, 500, -500, 1])
        x7, y7, z7 = np.matmul(P, [500, 0, -500, 1])
        x8, y8, z8 = np.matmul(P, [500, 500, -500, 1])
        
        # Drawing lines based on homogeneous points received
        # Top of the cube
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x5 / z5), int(y5 / z5)), (255, 0, 0), 2)
        cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x6 / z6), int(y6 / z6)), (255, 0, 0), 2)
        cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x7 / z7), int(y7 / z7)), (255, 0, 0), 2)
        cv2.line(frame, (int(x4 / z4), int(y4 / z4)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 2)

        # Bottom of the cube
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 255, 0), 2)
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x3 / z3), int(y3 / z3)), (0, 255, 0), 2)
        cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)), (0, 255, 0), 2)
        cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 255, 0), 2)

        # Sides of the cube
        cv2.line(frame, (int(x1 / z5), int(y5 / z5)), (int(x6 / z6), int(y6 / z6)), (0, 0, 255), 2)
        cv2.line(frame, (int(x2 / z5), int(y5 / z5)), (int(x7 / z7), int(y7 / z7)), (0, 0, 255), 2)
        cv2.line(frame, (int(x4 / z6), int(y6 / z6)), (int(x8 / z8), int(y8 / z8)), (0, 0, 255), 2)
        cv2.line(frame, (int(x5 / z7), int(y7 / z7)), (int(x8 / z8), int(y8 / z8)), (0, 0, 255), 2)

    # Showing the video to usee
        cv2.imshow("CUBE VIDEO", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
         break