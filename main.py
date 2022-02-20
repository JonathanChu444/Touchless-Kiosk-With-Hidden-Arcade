import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import time
import random

#Learned how to use Mediapipe Holistic and Pose estimation from Nicholas Renotte
#https://www.youtube.com/watch?v=pG4sUNDOZFg
#https://www.youtube.com/watch?v=06TE_U21FK4&t=1126s

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

counter = 0
stage = None

def calcDistance(a, b):
    a = np.array(a)
    b = np.array(b)
    p1 = (a[0] - b[0]) ** 2
    p2 = (a[1] - b[1]) ** 2
    length = np.sqrt(p1 + p2)
    return length

cv2.namedWindow('Touchless Kiosk System')
def nothing(x):
    print(x)
    pass


cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(3, width)
cap.set(4, height)
pyautogui.PAUSE = 0

prev_frame_time = 0
new_frame_time = 0
timer = 0
down = 1

pRWX, pRWY = 0, 0
cRWX, cRWY = 0, 0

secret_hand_state = 0

"""switch = 'Mode'
cv2.createTrackbar(switch, 'Touchless Kiosk System', 0, 2, nothing)"""

prev_frame_time = 0
new_frame_time = 0
prevv_frame_time = 0
neww_frame_time = 0
prevvv_frame_time = 0
newww_frame_time = 0
timer = 0
reactionTime = 0
rectangleHeight = 0
kneeStage = 1
count=0

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if (secret_hand_state == 0):
            try:
                if results.right_hand_landmarks:
                    right_hand_wrist = [results.left_hand_landmarks.landmark[0].x,
                                        results.left_hand_landmarks.landmark[0].y]
                    right_hand_index = [results.left_hand_landmarks.landmark[8].x,
                                        results.left_hand_landmarks.landmark[8].y]
                    right_hand_index_metacarpal = [results.left_hand_landmarks.landmark[5].x,
                                                   results.left_hand_landmarks.landmark[5].y]
                    right_hand_pinky_metacarpal = [results.left_hand_landmarks.landmark[17].x,
                                                   results.left_hand_landmarks.landmark[17].y]
                    right_hand_pinky = [results.left_hand_landmarks.landmark[20].x,
                                        results.left_hand_landmarks.landmark[20].y]
                    right_hand_thumb = [results.left_hand_landmarks.landmark[4].x,
                                        results.left_hand_landmarks.landmark[4].y]
                    # right_hand_middle = [results.left_hand_landmarks.landmark[12].x, results.left_hadn_landmarks.landmark[12].y]

                    right_hand_wristCoords = tuple(np.multiply(right_hand_wrist, [1920, 1080]).astype(int))
                    right_hand_indexCoords = tuple(np.multiply(right_hand_index, [1920, 1080]).astype(int))
                    right_hand_index_metacarpalCoords = tuple(
                        np.multiply(right_hand_index_metacarpal, [1920, 1080]).astype(int))
                    right_hand_pinky_metacarpalCoords = tuple(
                        np.multiply(right_hand_pinky_metacarpal, [1920, 1080]).astype(int))
                    right_hand_pinkyCoords = tuple(np.multiply(right_hand_pinky, [1920, 1080]).astype(int))
                    right_hand_thumbCoords = tuple(np.multiply(right_hand_thumb, [1920, 1080]).astype(int))

                    dist_RW_RI = calcDistance(right_hand_wristCoords, right_hand_indexCoords)
                    dist_RW_RPM = calcDistance(right_hand_wristCoords, right_hand_pinky_metacarpalCoords)
                    dist_RW_RIM = calcDistance(right_hand_wristCoords, right_hand_index_metacarpalCoords)
                    dist_RP_RIM = calcDistance(right_hand_pinkyCoords, right_hand_index_metacarpalCoords)
                    dist_RT_RIM = calcDistance(right_hand_thumbCoords, right_hand_index_metacarpalCoords)
                    dist_RPM_RIM = calcDistance(right_hand_pinky_metacarpalCoords, right_hand_index_metacarpalCoords)

                    """new_frame_time = time.time()
                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    timeCount = fps / fps
                    timeCount = int(timeCount)
                    timer += timeCount"""

                    xRW = right_hand_wristCoords[0]
                    yRW = right_hand_wristCoords[1]
                    """xRW1 = savgol_filter(xRW, 2239, 4)
                    yRW1 = savgol_filter(yRW, 1399, 4)"""
                    cRWX = pRWX + (xRW - pRWX) / 8
                    cRWY = pRWY + (yRW - pRWY) / 8

                    pyautogui.moveTo(cRWX, cRWY, duration=0.000001)
                    pRWX, pRWY = cRWX, cRWY

                    if (dist_RW_RI > 2 * dist_RW_RPM and down == 1):
                        pyautogui.leftClick(xRW, yRW)
                        down = 0
                    elif (dist_RW_RI < dist_RW_RPM):
                        down = 1

                    if (dist_RW_RIM < dist_RP_RIM):
                        pyautogui.scroll(12)
                    elif (dist_RPM_RIM < dist_RT_RIM):
                        pyautogui.scroll(-12)

                if results.left_hand_landmarks:
                    left_hand_index = [results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y]
                    left_hand_pinky = [results.right_hand_landmarks.landmark[20].x,
                                       results.right_hand_landmarks.landmark[20].y]
                    left_hand_wrist = [results.right_hand_landmarks.landmark[0].x,
                                       results.right_hand_landmarks.landmark[0].y]
                    left_hand_index_metacarpal = [results.right_hand_landmarks.landmark[5].x,
                                       results.right_hand_landmarks.landmark[5].y]
                    left_hand_pinky_metacarpal = [results.right_hand_landmarks.landmark[17].x,
                                       results.right_hand_landmarks.landmark[17].y]

                    left_hand_indexCoords = tuple(np.multiply(left_hand_index, [1920, 1080]).astype(int))
                    left_hand_pinkyCoords = tuple(np.multiply(left_hand_pinky, [1920, 1080]).astype(int))
                    left_hand_wristCoords = tuple(np.multiply(left_hand_wrist, [1920, 1080]).astype(int))
                    left_hand_index_metacarpalCoords = tuple(np.multiply(left_hand_index_metacarpal, [1920, 1080]).astype(int))
                    left_hand_pinky_metacarpalCoords = tuple(np.multiply(left_hand_pinky_metacarpal, [1920, 1080]).astype(int))

                    dist_LW_LI = calcDistance(left_hand_wristCoords, left_hand_indexCoords)
                    dist_LW_LP = calcDistance(left_hand_wristCoords, left_hand_pinkyCoords)
                    dist_LW_LIM = calcDistance(left_hand_wristCoords, left_hand_index_metacarpalCoords)
                    dist_LP_LIM = calcDistance(left_hand_pinkyCoords, left_hand_index_metacarpalCoords)
                    dist_LW_LPM = calcDistance(left_hand_wristCoords, left_hand_pinky_metacarpalCoords)

                    if ((dist_LW_LIM < dist_LP_LIM) and (dist_LW_LI > 2*dist_LW_LPM) and (945<= left_hand_wristCoords[0] <=975) and (525<= left_hand_wristCoords[1] <=555)):
                        secret_hand_state = 1
                    print(left_hand_wristCoords)

            except:
                print("didnt work")


        elif(secret_hand_state == 1 or secret_hand_state == 2 or secret_hand_state == 3):
            try:
                #landmarks = results.pose_landmarks.landmark
                #left_elbow = [results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y]

                left_wrist = [results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y]
                left_hip = [results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y]
                left_index = [results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y]
                left_pinky = [results.pose_landmarks.landmark[18].x, results.pose_landmarks.landmark[18].y]
                left_knee = [results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[26].y]

                right_wrist = [results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y]
                right_hip = [results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[23].y]
                right_index = [results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y]
                right_pinky = [results.pose_landmarks.landmark[17].x, results.pose_landmarks.landmark[17].y]
                right_knee = [results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[25].y]

                if results.left_hand_landmarks:
                    left_hand_index = [results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y]
                    left_hand_pinky = [results.right_hand_landmarks.landmark[20].x,
                                       results.right_hand_landmarks.landmark[20].y]
                    left_hand_wrist = [results.right_hand_landmarks.landmark[0].x,
                                       results.right_hand_landmarks.landmark[0].y]
                    left_hand_index_metacarpal = [results.right_hand_landmarks.landmark[5].x,
                                       results.right_hand_landmarks.landmark[5].y]
                    left_hand_pinky_metacarpal = [results.right_hand_landmarks.landmark[17].x,
                                       results.right_hand_landmarks.landmark[17].y]

                    left_hand_indexCoords = tuple(np.multiply(left_hand_index, [1920, 1080]).astype(int))
                    left_hand_pinkyCoords = tuple(np.multiply(left_hand_pinky, [1920, 1080]).astype(int))
                    left_hand_wristCoords = tuple(np.multiply(left_hand_wrist, [1920, 1080]).astype(int))
                    left_hand_index_metacarpalCoords = tuple(np.multiply(left_hand_index_metacarpal, [1920, 1080]).astype(int))
                    left_hand_pinky_metacarpalCoords = tuple(np.multiply(left_hand_pinky_metacarpal, [1920, 1080]).astype(int))

                    dist_LW_LI = calcDistance(left_hand_wristCoords, left_hand_indexCoords)
                    dist_LW_LP = calcDistance(left_hand_wristCoords, left_hand_pinkyCoords)
                    dist_LW_LIM = calcDistance(left_hand_wristCoords, left_hand_index_metacarpalCoords)
                    dist_LP_LIM = calcDistance(left_hand_pinkyCoords, left_hand_index_metacarpalCoords)
                    dist_LW_LPM = calcDistance(left_hand_wristCoords, left_hand_pinky_metacarpalCoords)
                    print(left_hand_wristCoords)

                    if ((dist_LW_LIM < dist_LP_LIM) and (dist_LW_LI > 2 * dist_LW_LPM) and (945<= left_hand_wristCoords[0] <=975) and (525<= left_hand_wristCoords[1] <=555)):
                        secret_hand_state = 1
                    elif ((dist_LW_LIM < dist_LP_LIM) and (945<= left_hand_wristCoords[0] <=975) and (525<= left_hand_wristCoords[1] <=555)):
                        secret_hand_state = 2
                    elif((dist_LW_LI > 2*dist_LW_LPM) and (945<= left_hand_wristCoords[0] <=975) and (525<= left_hand_wristCoords[1] <=555)):
                        secret_hand_state =3

                if (secret_hand_state == 1):
                    counter = 0
                    randX = 640
                    randY = 360
                    timeCount = 0
                    timeCountKnee = 0
                    timeCcountKnee = 0
                    maxHeight = 720
                    maxHeightt = 720
                    maxX = 1280
                    minX = 1230
                    randHeightt = 100
                    rectangleHeight = 0
                    randHeightFloor = 700
                    lol1 = 0
                    lol2 = 0
                    lol3 = 0
                    lol4 = 0
                # reaction timer
                elif (secret_hand_state == 2):
                    if ((((randX - 80 <= tuple(np.multiply(right_wrist, [1280, 720]).astype(int))[0] <= randX + 80) and (randY - 80 <= tuple(np.multiply(right_wrist, [1280, 720]).astype(int))[1] <= randY + 80)) or ((randX - 80 <=tuple(np.multiply(right_index, [1280, 720]).astype(int))[0] <= randX + 80) and (randY - 80 <= tuple(np.multiply(right_index, [1280, 720]).astype(int))[1] <= randY + 80)) or ((randX - 80 <=tuple(np.multiply(right_pinky,[1280,720]).astype(int))[0] <= randX + 80) and (randY - 80 <=tuple(np.multiply(right_pinky,[1280,720]).astype(int))[1] <= randY + 80))) or (((randX - 80 <= tuple(np.multiply(left_wrist, [1280, 720]).astype(int))[0] <= randX + 80) and (randY - 80 <= tuple(np.multiply(left_wrist, [1280, 720]).astype(int))[1] <= randY + 80)) or ((randX - 80 <= tuple(np.multiply(left_index, [1280, 720]).astype(int))[0] <= randX + 80) and (randY - 80 <= tuple(np.multiply(left_index, [1280, 720]).astype(int))[1] <= randY + 80)) or ((randX - 80 <=tuple(np.multiply(left_pinky,[1280,720]).astype(int))[0] <= randX + 80) and (randY - 80 <=tuple(np.multiply(left_pinky,[1280,720]).astype(int))[1] <= randY + 80)))):
                        randX = random.randrange(0, 1280, 10)
                        randY = random.randrange(80, 720, 10)
                        if (reactionTime < timer):
                            bestTime = timer
                        reactionTime = timer
                        timer = 0

                    new_frame_time = time.time()
                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    timeCount = fps / fps
                    timeCount = int(timeCount)

                    timer += timeCount

                    cv2.rectangle(image, (0, 0), (640, 73), (144, 118, 0), -1)
                    cv2.putText(image, 'Reaction Time', (15, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(reactionTime),
                                (10, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(timer),
                                (200, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(bestTime),
                                (390, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(image, (randX, randY), 80, (0, 215, 215), -1)
                    
                # jumpy death
                elif (secret_hand_state == 3):
                    neww_frame_time = time.time()
                    fpss = 1 / (neww_frame_time - prevv_frame_time)
                    prevv_frame_time = neww_frame_time
                    timeCcount = fpss / fpss
                    timeCcount = int(timeCcount)
                    timeCountKnee += timeCcount

                    cv2.rectangle(image, (0, 0), (400, 73), (144, 118, 0), -1)
                    cv2.putText(image, 'Ball Fall', (15, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(timeCountKnee),
                                (10, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(timeCcount),
                                (200, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    if ((rectangleHeight - 30 > randHeightt or maxX > 460) and rectangleHeight < randHeightFloor):
                        # cv2.circle(image, (400, rectangleHeight), 40, (0, 215, 255), -1)
                        if (((tuple(np.multiply(right_knee, [1280, 720]).astype(int))[1] <=
                              tuple(np.multiply(right_hip, [1280, 720]).astype(int))[
                                  1] + 30) and rectangleHeight >= 0)):
                            
                            kneeStage = 0
                        if (((tuple(np.multiply(left_knee, [1280, 720]).astype(int))[1] <=
                              tuple(np.multiply(left_hip, [1280, 720]).astype(int))[
                                  1] + 30) and rectangleHeight <= 700)):
                            kneeStage = 1

                        if (kneeStage == 0):
                            if (rectangleHeight < 10):
                                rectangleHeight = 0
                                randHeightFloor = random.randrange(maxHeight, minHeight, 25)
                            else:
                                rectangleHeight -= 13
                        elif (kneeStage == 1):
                            if (rectangleHeight >= 700):
                                rectangleHeight = 700
                                randHeightFloor = random.randrange(maxHeight, minHeight, 25)
                            else:
                                rectangleHeight += 13
                        if (maxX >= 400):
                            minX = maxX + 35
                            maxX -= 10
                            #randHeightFloor = random.randrange(maxHeight, minHeight, 25)
                            # randxX = random.randrange(minX, maxX, 20)
                        elif (maxX < 400):
                            minX = 1280
                            maxX = 1230
                            if (maxHeight >= 540):
                                minHeight = maxHeight
                                maxHeight -= 50
                                #randHeightFloor = random.randrange(maxHeight, minHeight, 25)
                            elif (maxHeight <= 540):
                                maxHeight = 700
                            randHeightt = random.randrange(200, 400, 40)
                            timeCountKnee = 0
                    else:
                        cv2.putText(image, 'GAME OVER', (300, 360),
                                    cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(image, (400, rectangleHeight), 40, (0, 215, 255), -1)
                    cv2.rectangle(image, (maxX, 0), (minX, randHeightt), (0, 0, 255), -1)
                    cv2.rectangle(image, (0, randHeightFloor), (1280, 720), (0, 0, 255), -1)
            except:
                pass
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Touchless Kiosk System', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()