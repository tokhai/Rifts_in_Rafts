import os
import numpy as np
import cv2
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import sys
import matplotlib.pyplot as plt
import yaml
from skimage.morphology import watershed, remove_small_objects
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border

channelwidth = 52980. #channel width in um

########Initial setup and config parsing###########

with open(sys.argv[1], 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

inpath  = cfg['Video']['inpath']
videoname  = cfg['Video']['videoname']
foldername = inpath + videoname
width  = cfg['Video']['width']
fps  = cfg['Video']['fps']
xlow = cfg['Video']['xlow']
xupper = cfg['Video']['xupper']
ylow = cfg['Video']['ylow']
yupper = cfg['Video']['yupper']
velocity = cfg['Video']['velocity']
nframe = cfg['Video']['nframe']

blurdia = cfg['ImageProcessing']['blurdia']
blurthre = cfg['ImageProcessing']['blurthre']
particlethre = cfg['ImageProcessing']['particlethre']

Ly = yupper-ylow
lowerdia = 500./(channelwidth/(yupper-ylow))
upperdia = 710./(channelwidth/(yupper-ylow))
scale = float(channelwidth)/float(yupper-ylow) #px to um scale

filename0 = foldername+'/crop/{:04d}.png'
filename1 = foldername+'/raw/bd-{}.png'
filename5 = foldername+'/raw/paroriginal-{}.png'

########Read the initial packing##########
binimg0 = cv2.bitwise_not(cv2.imread(filename5.format(0), 0))
labels = label(clear_border(binimg0))
props = regionprops_table(labels, properties=['area'])
data = pd.DataFrame(props)
area = data['area'].to_numpy() #find all initial holes
hole_max = (np.sqrt(3)/4-np.pi/8)*upperdia*upperdia #remove holes bounded by four largest particles
areacut = area[area>hole_max]
areatightcut = area[area>6] #remove noise
avg = np.mean(areatightcut) #compute the average size of the holes in the initial packing
smallarea = int(np.ceil(avg))
avgemptysize = np.sqrt(avg)
s = int(np.ceil(avgemptysize/2)) #determine the dilation size
print(s)
M = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*s+1,2*s+1))

for i in range(0, nframe):
    rawimg = cv2.imread(filename5.format(i), 0)
    bdimg = cv2.imread(filename1.format(i), 0)
    binimg = cv2.bitwise_and(rawimg, bdimg)

    #close the holes using the dilation matrix obtained above
    preproc = 255 - cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, M)
    arr = preproc > 0
    cleaned = remove_small_objects(arr, min_size=smallarea)
    dilresult = 255 - 255*cleaned.astype(np.uint8)
    
    #measure every horizonal slices
    dilresult[:, :1] = 0
    dilresult[:, -1:] = 0
    inv_16 = dilresult.astype(np.int16)/255
    linediff = np.diff(inv_16)
    startpts = np.where(linediff==1)
    endpts = np.where(linediff==-1)
    result = endpts[1]-startpts[1]

    #print out the average of all the measurements
    print(np.mean(result))
