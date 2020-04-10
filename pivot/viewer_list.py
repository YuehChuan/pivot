# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:05:34 2020

@author: shooter
"""

import matplotlib.pyplot as plt
#from InfoGenerator import prepareImg
import cv2 as cv
import numpy as np

  
def genSteps(steps):
  interval = steps-1
  MIN = 0
  MAX = 255
  intervalValue = (MAX - MIN)/interval
  step = []
  for idx in range(0, interval):
    if idx == 0:
      step.append((int(MIN), int(intervalValue-1)))
    elif idx == interval-1:
      step.append((int(idx * intervalValue), int(MAX)))
    else:
      step.append((int(idx*intervalValue), int((idx+1)*intervalValue)))
  return step

def clipByStep(val, step):
  retValue = 0
  for s in step:
    if s[0] <= val <= s[1]:
      if val <= (s[1] + s[0])/2:
        retValue = s[0]
      else:
        retValue = s[1]
  
  return retValue

def expandAmplitude(img):
  img = img.astype(np.float32)/255.0
  minVal, maxVal = np.amin(img), np.amax(img)
  img = (img - minVal) / (maxVal - minVal) * 255.0
  img = img.astype(np.uint8)
  return img
    

def clipToSteps(Img, steps=30):
  step = genSteps(steps)
  print(step)
  h = Img.shape[0]
  w = Img.shape[1]
  for y in range(0, h):
    for x in range(0, w):
      Img[y, x] = clipByStep(Img[y, x], step)  

  return Img

def Viewer(PATH:str):
  # import image
  # PATH = './test/10.jpg'
  #PATH = 'medBlur.jpg'
  Img = cv.imread(PATH, 0)
  # Img, bodyDir, handleDir, target = prepareImg(Img)
  Img = cv.medianBlur(Img, 15)
  Img = expandAmplitude(Img)
  Img = clipToSteps(Img)

  # Img[Img<25] = 0
  # Img[Img>200] = 255
  # select col
  selectCol = 250
  Col = Img[:, selectCol]
  x = []
  y = []
  for idx, val in enumerate(Col):
    x.append(idx)
    y.append(val)

  # draw line
  # Img[:,selectCol] = 255
  #plt.plot(x, y)
  #plt.show()
  #plt.imshow(Img, cmap='gray')
  #plt.title("result"), plt.xticks([]), plt.yticks([])
  #plt.show()

  # select row
  '''
  selectRow = 300
  Row = Img[selectRow,:]
  x = []
  y = []

  for idx, val in enumerate(Row):
      x.append(idx)
      y.append(val)

  Img[selectRow,:] = 255

  plt.plot(x, y)
  plt.show()
  plt.imshow(Img,cmap = 'gray')
  plt.title("result"), plt.xticks([]), plt.yticks([])
  plt.show()
  '''
  Col_np = np.asarray(Col, dtype=np.float64)

  return Col_np
