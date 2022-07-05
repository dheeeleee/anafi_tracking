"""
it computes a vector of the ball's direction from the center of the
screen. The axes are shown below :
+y                 


Y                   (0,0)               


-Y                 
-X                    X                    +X

Based on the tutorial:
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

"""

import time
import cv2

class HSV:
    """
    A basic color tracker, it will look for colors in a range and
    create an x and y offset valuefrom the midpoint
    """

    def __init__(self, height, width, color_lower, color_upper):
        self.color_lower = color_lower
        self.color_upper = color_upper
        self.midx = int(width / 2)
        self.midy = int(height / 2)
        self.xoffset = 0
        self.yoffset = 0
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def draw_arrows(self, frame):
        """Show the direction vector output in the cv2 window"""
        #cv2.putText(frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.arrowedLine(frame, (self.midx, self.midy),
                        (self.midx + self.xoffset, self.midy - self.yoffset),
                        (0, 0, 255), 5)
        return frame

    def detect(self, frame):
        """Simple HSV color space tracking"""
        # resize the frame, blur it, and convert it to the HSV
        # color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            # ((x, y), radius) = cv2.minEnclosingCircle(c)
            self.x, self.y, self.w, self.h = cv2.boundingRect(c)
            

            # M = cv2.moments(c)
            # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if self.w > 10:
                pass
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                # cv2.circle(frame, (int(x), int(y)), int(radius),
                #            (0, 255, 255), 2)
                # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # draw bounding box on the frame
                # cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 255), 2)

                # self.xoffset = int(center[0] - self.midx)
                # self.yoffset = int(self.midy - center[1])
            # else:
            #     self.xoffset = 0
            #     self.yoffset = 0
            else : 
                self.x = 0
                self.y = 0
                self.w = 0
                self.h = 0
        else:
            # self.xoffset = 0
            # self.yoffset = 0
            self.x = 0
            self.y = 0
            self.w = 0
            self.h = 0

        return  self.x, self.y, self.w, self.h
