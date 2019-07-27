import cv2
import numpy as np
import os
import re

os.chdir('../../')
crowd11_folder = 'Data/Crowd-11/'
crowd11_of_folder = 'Data/Crowd11_OpticalFlow/'
list_videos = os.listdir(crowd11_folder)
print(list_videos)
for video_name in list_videos:
    cap = cv2.VideoCapture(crowd11_folder+video_name)
    ret, frame1 = cap.read()
    previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.DualTVL1OpticalFlow_create()


    flows = list()
    cap_bool = True
    while(cap_bool):
        ret, frame2 = cap.read()
        if ret == False:
            cap_bool = False
            continue
        print(ret, frame2.shape)
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = optical_flow.calc(previous_frame, next_frame, None)
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(flow.shape)
        flows.append(flow)
        previous_frame = next_frame

    flow_filename = re.findall("(.*?)\.[ma][pv][4i]", video_name)[0]
    video_flows = np.asarray(flows)
    np.save(crowd11_of_folder+flow_filename, video_flows)

    cap.release()