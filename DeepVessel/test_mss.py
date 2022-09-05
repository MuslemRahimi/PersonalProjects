import time

import cv2
import mss
import numpy
import pyautogui


num = 1
with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 65, "left": 65, "width": 1024, "height": 576}
    #monitor = sct.monitors[2]
    
    while "Screen capturing":
        last_time = time.time()
        '''
        num = num+1
        if num % 100 ==0:
            pyautogui.press('x')
        '''
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Display the picture
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        #edges = cv2.Canny(blurred, 50, 100)
        edges = cv2.Canny(blurred, 50,50)



        cv2.imshow("OpenCV/Numpy normal", edges)

        # Display the picture in grayscale
        #cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        #print("fps: {}".format(1 / (time.time() - last_time)))


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break