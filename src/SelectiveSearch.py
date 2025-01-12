import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
import time

#Init
x = 0.08
time_elapsed = 0.1
unknown_count = 0
start = 0
k = 4
fac = 0.333
video = 'atag.mp4'

mtx = np.array([[818.53308038, 0., 330.99241018],
                   [0., 816.61728871, 232.94501515],
                    [0., 0., 1.]])
dist = np.array([[0.02188201, 0.40525476, -0.01361047, 0.00440778, -0.29207344]])
worldPoints = np.array([(-x, x, 0), (x, x, 0), (x, -x, 0), (-x, -x, 0)])
worldPoints = worldPoints.astype('float32')
imgPtsMat = np.array([(0, 0), (0, 0), (0, 0), (0, 0)])
imgPtsMat = imgPtsMat.astype('float32')
accuracy = 1

frame_count = 0

n = 1

file = open('xy_data2.txt', 'w')

detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

#Tracking Class
class TrackTag:

  def __init__(self):
    self.prevPos = []
    self.prevTime = []

  def reset_tag(self):
    self.prevPos = []
    self.prevTime = []

  def get_position(self):
    return self.prevPos[len(self.prevPos)-1]

  def update(self, a, b, c, d, t):
    curPos = corners_to_position(a, b, c, d)
    self.prevPos.append(curPos)
    self.prevTime.append(t)
    if (len(self.prevPos) > n):
        self.prevPos.pop(0)
        self.prevTime.pop(0)

  def predict(self, t):
    matPos = np.array(self.prevPos)
    nextPos = []
    matTime = []
    for i in self.prevTime:
        matTime.append([i*i, i, 1])
    if matPos.size > 0:
        for j in range(0, 6):
            col = matPos[:, j]
            coef = (np.linalg.lstsq(matTime, col, rcond=None))[0]
            val = coef[0]*t*t+coef[1]*t+coef[2]
            nextPos.append(val)
    else:
        nextPos = [0, 0, 0, 0, 0, 0]
    return format_corners(position_to_corners(nextPos))

#Corners to position
def corners_to_position(a, b, c, d):
  imagePoints = np.array([(a), (b), (c), (d)])
  imagePoints = imagePoints.astype('float32')
  found,rvec,tvec = cv2.solvePnP(worldPoints, imagePoints, mtx, dist)
  rotation_matrix = (cv2.Rodrigues(rvec))[0]
  R = rotation_matrix.transpose()
  xyz = -R @ tvec

  r =  Rotation.from_matrix(rotation_matrix)
  angles = r.as_euler("xyz",degrees=True)

  return np.array([xyz[0][0], xyz[1][0], xyz[2][0], angles[0], angles[1], angles[2]])

#Position to corners
def position_to_corners(pose):
  angles = (pose[3], pose[4], pose[5])
  # print(angles)
  r = Rotation.from_euler("xyz",angles,degrees=True)
  rotation_matrix = r.as_matrix()
  rvec = (cv2.Rodrigues(rotation_matrix))[0]

  xyz = np.array([[pose[0]],
                  [pose[1]],
                  [pose[2]]])
  R = rotation_matrix
  tvec = R @ -xyz
  corners, a = cv2.projectPoints(worldPoints, rvec, tvec, mtx, dist, imgPtsMat)
  return corners

#Format Corner data
def format_corners(corners):
  ptA = (capNum(int(corners[0][0][0])), capNum(int(corners[0][0][1])))
  ptB = (capNum(int(corners[1][0][0])), capNum(int(corners[1][0][1])))
  ptC = (capNum(int(corners[2][0][0])), capNum(int(corners[2][0][1])))
  ptD = (capNum(int(corners[3][0][0])), capNum(int(corners[3][0][1])))
  return ptA, ptB, ptC, ptD

def capNum(num):
  if num > 10000:
    return 10000
  if num < -10000:
    return -10000
  else:
    return num

#Intersection / Union
def intersection_over_union(q1, q2):
    poly1 = Polygon(q1)
    poly2 = Polygon(q2)
    if poly1.intersects(poly2): 
        intersect = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersect/union
    return 0

def scale_polygon(vertices, scale_factor):
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    center_x = sum(x) / len(x)
    center_y = sum(y) / len(y)
    max_x = 0
    min_x = 10000
    max_y = 0
    min_y = 10000
    for i in range(len(vertices)):
        new_x = (vertices[i][0] - center_x) * scale_factor + center_x
        if new_x > max_x:
          max_x = new_x
        if new_x < min_x:
          min_x = new_x
        new_y = (vertices[i][1] - center_y) * scale_factor + center_y
        if new_y > max_y:
          max_y = new_y
        if new_y < min_y:
          min_y = new_y
    new_vertices = [(int(min_x), int(min_y)), (int(max_x), int(min_y)), (int(max_x), int(max_y)), (int(min_x), int(max_y))]
    return new_vertices[0], new_vertices[1], new_vertices[2], new_vertices[3]


Tag = TrackTag()

cap = cv2.VideoCapture(video)

startTot = time.time()
totAcc = 0
while True:
    ret, frame = cap.read()
    if (not ret):
        endTot = time.time()
        break

    results = []

    end = time.time()
    time_elapsed =  1
    start = time.time()

    preA, preB, preC, preD = Tag.predict(time_elapsed)

    adjA, adjB, adjC, adjD = scale_polygon([preA, preB, preC, preD], (1+k*(1-accuracy)))

    #cv2.line(frame, adjA, adjB, (255, 255, 255), 4)
    #cv2.line(frame, adjB, adjC, (255, 255, 255), 4)
    #cv2.line(frame, adjC, adjD, (255, 255, 255), 4)
    #cv2.line(frame, adjD, adjA, (255, 255, 255), 4)

    cropped_frame = frame[adjA[1]:adjC[1], adjA[0]:adjC[0]]

    if cropped_frame.size != 0:
        cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(cropped)
    
    x_o = adjA[0]
    y_o = adjA[1]
        
    if results != []:
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0])+x_o, int(ptB[1])+y_o)
            ptC = (int(ptC[0])+x_o, int(ptC[1])+y_o)
            ptD = (int(ptD[0])+x_o, int(ptD[1])+y_o)
            ptA = (int(ptA[0])+x_o, int(ptA[1])+y_o)

            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

            Tag.update(ptA, ptB, ptC, ptD, time_elapsed)

            accuracy = intersection_over_union([ptA, ptB, ptC, ptD], [preA, preB, preC, preD])
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

            Tag.update(ptA, ptB, ptC, ptD, time_elapsed)

            accuracy = intersection_over_union([ptA, ptB, ptC, ptD], [preA, preB, preC, preD])
    
    cv2.line(frame, adjA, adjB, (255, 0, 0), 4)
    cv2.line(frame, adjB, adjC, (255, 0, 0), 4)
    cv2.line(frame, adjC, adjD, (255, 0, 0), 4)
    cv2.line(frame, adjD, adjA, (255, 0, 0), 4)
    #cv2.line(frame, preA, preB, (255, 0, 0), 4)
    #cv2.line(frame, preB, preC, (255, 0, 0), 4)
    #cv2.line(frame, preC, preD, (255, 0, 0), 4)
    #cv2.line(frame, preD, preA, (255, 0, 0), 4)
    frame_count +=1
    if (frame_count==59):
        if (frame_count == 59):
            cv2.imwrite(('frame' + str(frame_count) + '.png'), frame)
    cv2.imshow('frame', frame)
    
    totAcc += accuracy
    if cv2.waitKey(1) == ord('q'):
        endTot = time.time()
        break
file.close()
print(frame_count / (endTot - startTot))
print(totAcc / frame_count)
cap.release()
cv2.destroyAllWindows()




