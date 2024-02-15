import numpy as np 
import cv2 
import time
import sys 
import os
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import airsim
import Model as Net 


from follow import distance

#create an instance of the distance class
#import airsim





intialTracbarVals = [179,196,206,386]

#callback function for the trackbars in cv function
def nothing(a):
    pass
def initializeTrackbars(intialTracbarVals,wT=497, hT=245):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)
def valTrackbars(wT=497, hT=245):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points



if __name__ == '__main__':
    display=1
    cap = cv2.VideoCapture("4.mp4") #path to the video file shd be given here
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]
    initializeTrackbars(intialTracbarVals)
    pallete = [[128, 64, 128],
               [244, 35, 232],
               [70, 70, 70],
               [102, 102, 156],
               [190, 153, 153],
               [153, 153, 153],
               [250, 170, 30],
               [220, 220, 0],
               [107, 142, 35],
               [152, 251, 152],
               [70, 130, 180],
               [220, 20, 60],
               [255, 0, 0],
               [0, 0, 142],
               [0, 0, 70],
               [0, 60, 100],
               [0, 80, 100],
               [0, 0, 230],
               [119, 11, 32],
               [0, 0, 0]]

    model = Net.ESPNet(20, 2, 5)
    model.load_state_dict(torch.load(os.path.expanduser("~/ESPNet/pretrained/decoder/espnet_p_2_q_5.pth"),
                                     map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model = model.cuda() if torch.cuda.is_available() else model
    client = airsim.MultirotorClient()
    wT=497
    hT=245
    video_path = 'home'   #change the path to your video source 
    cap = cv2.VideoCapture(video_path)
while True:
    ret,frame = cap.read()
    if not ret:
        print("No video found please check the video source")
        break  #if there is no video or video feed just break the loop
    image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
    if image_response is not None and len(image_response):
        image_response_np = np.frombuffer(image_response[0].image_data_uint8, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_response_np, cv2.IMREAD_COLOR)

        img_orig = np.copy(image_cv2)
        img = image_cv2.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
            img[:, :, j] /= std[j]\

    img = cv2.resize(img, (1024, 512))
    # img_orig = cv2.resize(img_orig, (1024, 512))
    img = img.astype(np.float32) / 255.0
    # img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    img_variable = Variable(img_tensor)
    img_variable = img_variable.cuda() if torch.cuda.is_available() else img_variable

    img_out = model(img_variable)



