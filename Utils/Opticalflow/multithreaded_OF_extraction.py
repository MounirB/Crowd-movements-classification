import threading as th
import cv2
import numpy as np
import os
import re
import random as rd

class myThread(th.Thread):
   def __init__(self, name, threadID, chunk):
      th.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.chunk = chunk
   def run(self):
      print("Starting " + self.name)
      extract_opticalflow(self.name, self.chunk)

def chunkIt(seq, num):
   """
   Split a list into num parts
   :param seq: The list to split
   :param num: Number of parts to split
   :return: A list of num sublists
   """
   avg = len(seq) / float(num)
   out = []
   last = 0.0

   while last < len(seq):
      out.append(seq[int(last):int(last + avg)])
      last += avg

   return out

def extract_opticalflow(threadName, list_videos):
   """
   Launch a thread that extracts optical flow
   :param threadName: The name of the thread
   :param list_videos: The list of videos the thread treats
   :return: Creates the optical flow for each video
   """
   for video_name in list_videos:
      cap = cv2.VideoCapture(crowd11_folder + video_name)
      ret, frame1 = cap.read()
      previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
      optical_flow = cv2.DualTVL1OpticalFlow_create()

      flows = list()
      cap_bool = True
      while (cap_bool):
         ret, frame2 = cap.read()
         if ret == False:
            cap_bool = False
            continue
         print(threadName, ret, frame2.shape)
         next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
         flow = optical_flow.calc(previous_frame, next_frame, None)
         flows.append(flow)
         previous_frame = next_frame

      flow_filename = re.findall("(.*?)\.[ma][pv][4i]", video_name)[0]
      video_flows = np.asarray(flows)
      np.save(crowd11_of_folder + flow_filename, video_flows)

      cap.release()

if __name__ == '__main__':
   os.chdir('../../')
   crowd11_folder = 'Data/Crowd-11/'
   crowd11_of_folder = 'Data/Crowd11_OpticalFlow/'
   list_videos = os.listdir(crowd11_folder)
   nb_threads = 10

   chunks_videos = chunkIt(list_videos, nb_threads)

   threads = []
   for num_thread in range(0, nb_threads):
      # Create new thread
      thread = myThread("Thread-"+str(num_thread), num_thread, chunks_videos[num_thread])
      # Start new Thread
      thread.start()
      # Add thread to thread list
      threads.append(thread)

   # Wait for all threads to complete
   for thread in threads:
      thread.join()
   print("Exiting main thread")
