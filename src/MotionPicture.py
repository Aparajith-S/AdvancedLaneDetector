import cv2
import LaneDetector
import numpy as np
#cap = cv2.VideoCapture("../challenge_video.mp4")
cap = cv2.VideoCapture("../project_video.mp4")
out = cv2.VideoWriter("../output_video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280,720))
while not cap.isOpened():
    cap = cv2.VideoCapture("../project_video.mp4")
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
failcounter=0
old_pos_frame=0
obj = LaneDetector.LaneFinder()
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        # cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
        if(old_pos_frame != pos_frame):
            failcounter=0
        result = obj.iterate(frame)
        x_midpoint = frame.shape[1]/2
        Lx=int(obj.left.bestx[-1])
        Rx=int(obj.right.bestx[-1])
        EgoPos='right'
        print(x_midpoint)
        EgoPosVal = ((x_midpoint-Lx) + (x_midpoint-Rx))
        EgoPos = str(round(np.absolute(EgoPosVal * 3.7 / 910), 2))
        perf=['Low','High']
        if EgoPosVal > 0 :
            EgoPos += '[m] to the right'
        else:
            EgoPos += '[m] to the left'
        str_cur = "Radius of Curvature = {}[m]".\
                      format(int(np.min([obj.left.radius_of_curvature, obj.right.radius_of_curvature]))) \
                  +"performance:"+perf[obj.once]
        #find position of car (look for C value of the polynomial!) gives the x position of lanes at sides of Ego (y=0)
        cv2.putText(result, str_cur, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(result, str("Ego vehicle is ")+EgoPos, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('processed', result)
        out.write(result)
    else:
        failcounter+=1
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)
        #failure to read 5 consecutive times will terminate
        if failcounter==5:
            break
        old_pos_frame = pos_frame
    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
out.release()
cv2.destroyAllWindows()