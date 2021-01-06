import numpy as np
from Polyfit import finalPipeline,Line
import cv2

class LaneFinder():
    def __init__(self):
        self.left = Line()
        self.right = Line()
        #a variable to switch to fast lane searching
        self.once = 0
        self.Minv = []
    def iterate(self, img):
        self.Minv,self.left,self.right,ploty,self.once,warped = finalPipeline(img,self.left,self.right,self.once)
        #if(np.absolute(self.left.radius_of_curvature-self.right.radius_of_curvature) > 500):
        #   self.once=0
        result = displayLane(warped,img,self.Minv,ploty,self.left,self.right)
        return result
    def reset(self):
        once=0

#input the undistorted image, warped image, inverse perspective transform Matrix
def displayLane(warped,undist,Minv,ploty,left=Line(),right=Line()):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

if __name__ == "__main__":
    image = cv2.imread("../test_images/straight_lines1.jpg")
    # View output
    obj = LaneFinder()
    #just run twice to see switch over to search algorithm
    obj.iterate(image)
    obj.iterate(image)
    cv2.waitKey()
    cv2.destroyAllWindows()