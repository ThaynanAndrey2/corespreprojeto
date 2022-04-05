# corespreprojeto
Detecção de Cores - Pré Projeto (Deep Learning)
import numpy as np
import imutils
import cv2
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red': (0, 50, 120),
'blue': (90, 60, 0),
'yellow': (25, 70, 120),
'green': (40, 70, 80)}
upper = {'red': (10, 255, 255),
'blue': (121, 255, 255),
'yellow': (30, 255, 255),
'green': (70, 255, 255)}
# define standard colors for circle around the object
colors = {'red': (0, 0, 255),
'blue': (255, 0, 0),
'yellow': (0, 255, 217),
'green': (0, 128, 0)}

#INICIANDO CAPTURA EM TEMPO REAL
camera = cv2.VideoCapture(0)

while True:
    # read the video in real time
    _, frame = camera.read()
    # resize the frame
    frame = imutils.resize(frame, width=1000)
    #(H, W) = frame.shape[:2]
    # convert it to the HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # for each color in dictionary check object in frame
    # for key, value in upper.items():
    kernel = np.ones((9, 9), np.uint8)
    # creating mask for red color
    maskRed = cv2.inRange(hsv, lower['red'], upper['red'])
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernel)
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel)
    # creating mask for blue color
    maskBlue = cv2.inRange(hsv, lower['blue'], upper['blue'])
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel)
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel)
    # creating mask for yellow color
    maskYellow = cv2.inRange(hsv, lower['yellow'], upper['yellow'])
    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel)
    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel)
    # creating mask for green color
    maskGreen = cv2.inRange(hsv, lower['green'], upper['green'])
    maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernel)
    maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_CLOSE, kernel)
    
    # find contours on the masks and initialize the currents
    cntsRed = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsBlue = cv2.findContours(maskBlue.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsYellow = cv2.findContours(maskYellow.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsGreen = cv2.findContours(maskGreen.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # searching for red color
    if len(cntsRed) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        threshold = 0.5
        c = max(cntsRed, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > threshold:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)),
            int(radius), colors['red'], 2)
            cv2.putText(frame, 'red' + " object", (int(x-radius), int(y - radius)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['red'], 2)
    
    # searching for blue color
    if len(cntsBlue) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        threshold = 0.5
        c = max(cntsBlue, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > threshold:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)),
            int(radius), colors['blue'], 2)
            cv2.putText(frame, 'blue' + " object", (int(x-radius), int(y - radius)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['blue'], 2)

    # searching for yellow color
    if len(cntsYellow) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        threshold = 0.5
        c = max(cntsYellow, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > threshold:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)),
            int(radius), colors['yellow'], 2)
            cv2.putText(frame, 'yellow' + " object", (int(x-radius), int(y - radius)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['yellow'], 2)

    # searching for green color
    if len(cntsGreen) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        threshold = 0.5
        c = max(cntsGreen, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > threshold:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)),
            int(radius), colors['green'], 2)
            cv2.putText(frame, 'green' + " object", (int(x-radius), int(y - radius)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['green'], 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
