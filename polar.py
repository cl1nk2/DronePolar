# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np
import polanalyser as pa

import pandas as pd
from os.path import exists

from svgutils.compose import *
from scour import scour
import os

def cvtStokesToDoLP(stokes_vectors):
  S0 = stokes_vectors[..., 0]
  S1 = stokes_vectors[..., 1]
  S2 = stokes_vectors[..., 2]
  return np.sqrt(S1**2+S2**2)/(S0+(S0==0))
  
def detectWater(img_int, img_poldeg, img_poldir, img_overexposed, img_underexposed):
    img_water = img_int.copy()
    
    img_poldeg_cp = 1-img_poldeg.copy()
    img_poldir_cp = 2*img_poldir.copy()
    
    threshold_poldeg_low = 0.15
    threshold_poldir_range = 10*np.pi/180
    
    img_poldeg_above_threshold = (img_poldeg_cp >= threshold_poldeg_low)
    img_poldir_in_threshold = (((img_poldir <= 0*np.pi/180 + threshold_poldir_range) & (img_poldir >= 0*np.pi/180 - threshold_poldir_range)) | ((img_poldir <= 180*np.pi/180 + threshold_poldir_range) & (img_poldir >= 180*np.pi/180 - threshold_poldir_range)))
    
    img_water_detect = img_poldeg_above_threshold & img_poldir_in_threshold
    
    img_water = cv2.cvtColor(img_water, cv2.COLOR_RGB2GRAY)
    img_water = cv2.cvtColor(img_water, cv2.COLOR_GRAY2RGB)
    
    img_water[img_water_detect] = [255, 0, 0]
    
    img_water[img_overexposed] = [255, 255, 255]
    img_water[img_underexposed] = [255, 255, 255]
    
    return img_water

def applyColorToDoLP(img_DoLP, img_overexposed, img_underexposed):
  img_DoLP = 255*(1-img_DoLP)
  
  img_DoLP = cv2.cvtColor(img_DoLP.astype('float32'), cv2.COLOR_GRAY2RGB)
  
  img_DoLP[img_overexposed] = [0, 0, 255]
  img_DoLP[img_underexposed] = [255, 0, 0]
  
  return img_DoLP

def applyAlgonetColorToAoLP(img_AoLP, img_overexposed, img_underexposed):
  img_AoLP = img_AoLP*255/np.pi
  img_AoLP = cv2.cvtColor(img_AoLP.astype('float32'), cv2.COLOR_GRAY2RGB)

  for i in range(img_AoLP.shape[0]):
    for j in range(img_AoLP.shape[1]):
      pixel = img_AoLP.item(i, j, 0)
      step = 64
      if pixel == 0.0:
        pass
      elif pixel <= step:
        img_AoLP.itemset(i, j, 0, 0)
        img_AoLP.itemset(i, j, 1, (1-pixel/step/2)*255)
        img_AoLP.itemset(i, j, 2, 0)
      elif pixel > step and pixel <= step*2:
        img_AoLP.itemset(i, j, 0, 0)
        img_AoLP.itemset(i, j, 1, 0)
        img_AoLP.itemset(i, j, 2, ((pixel-step)/step/2+0.5)*255)
      elif pixel > step*2 and pixel <= step*3:
        img_AoLP.itemset(i, j, 0, 0)
        img_AoLP.itemset(i, j, 1, (1-((pixel-step*2)/step/2))*255)
        img_AoLP.itemset(i, j, 2, (1-((pixel-step*2)/step/2))*255)
      elif pixel > step*3 and pixel <= step*4:
        img_AoLP.itemset(i, j, 0, ((pixel-step*3)/step/2+0.5)*255)
        img_AoLP.itemset(i, j, 1, 0)
        img_AoLP.itemset(i, j, 2, 0)
        
  img_AoLP[img_overexposed] = [255, 255, 255]
  img_AoLP[img_underexposed] = [255, 255, 255]
        
  return img_AoLP

def imwrite_result(img_in, name):
  width = int(img_in[1].shape[1]*0.25)
  height = int(img_in[1].shape[0]*0.25)
  dim = (width, height)

  img_0  = cv2.resize(img_in[0], dim, interpolation = cv2.INTER_AREA)
  img_0  = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
  img_00 = cv2.resize(img_in[1], dim, interpolation = cv2.INTER_AREA)
  img_00 = cv2.rotate(img_00, cv2.ROTATE_90_CLOCKWISE)
  img_01 = cv2.resize(img_in[2], dim, interpolation = cv2.INTER_AREA)
  img_01 = cv2.rotate(img_01, cv2.ROTATE_90_CLOCKWISE)
  img_02 = cv2.resize(img_in[3], dim, interpolation = cv2.INTER_AREA)
  img_02 = cv2.rotate(img_02, cv2.ROTATE_90_CLOCKWISE)
  img_10 = cv2.resize(img_in[4], dim, interpolation = cv2.INTER_AREA)
  img_10 = cv2.rotate(img_10, cv2.ROTATE_90_CLOCKWISE)
  img_11 = cv2.resize(img_in[5], dim, interpolation = cv2.INTER_AREA)
  img_11 = cv2.rotate(img_11, cv2.ROTATE_90_CLOCKWISE)
  img_12 = cv2.resize(img_in[6], dim, interpolation = cv2.INTER_AREA)
  img_12 = cv2.rotate(img_12, cv2.ROTATE_90_CLOCKWISE)
  img_20 = cv2.resize(img_in[7], dim, interpolation = cv2.INTER_AREA)
  img_20 = cv2.rotate(img_20, cv2.ROTATE_90_CLOCKWISE)
  img_21 = cv2.resize(img_in[8], dim, interpolation = cv2.INTER_AREA)
  img_21 = cv2.rotate(img_21, cv2.ROTATE_90_CLOCKWISE)
  img_22 = cv2.resize(img_in[9], dim, interpolation = cv2.INTER_AREA)
  img_22 = cv2.rotate(img_22, cv2.ROTATE_90_CLOCKWISE)

  img_black = cv2.cvtColor(np.zeros([height, width]).astype('float32'), cv2.COLOR_GRAY2RGB)
  img_black = cv2.rotate(img_black, cv2.ROTATE_90_CLOCKWISE)

  img_h0 = cv2.hconcat([img_black, img_0, img_black])
  img_h1 = cv2.hconcat([img_00, img_01, img_02])
  img_h2 = cv2.hconcat([img_10, img_11, img_12])
  img_h3 = cv2.hconcat([img_20, img_21, img_22])

  img = cv2.vconcat([img_h0, img_h1, img_h2, img_h3])
  
  name = name[:-4] + "_eval.png"

  cv2.imwrite(name, img)

def imwrite_result_split(img_in, name):
  name = name[:-4]
  width = int(img_in[1].shape[1]*0.25)
  height = int(img_in[1].shape[0]*0.25)
  dim = (width, height)

  img_int  = cv2.resize(img_in[0], dim, interpolation = cv2.INTER_AREA)
  img_int  = cv2.rotate(img_int, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_int.png", img_int)
  
  img_int_r = cv2.resize(img_in[1], dim, interpolation = cv2.INTER_AREA)
  img_int_r = cv2.rotate(img_int_r, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_int_r.png", img_int_r)
  
  img_int_g = cv2.resize(img_in[2], dim, interpolation = cv2.INTER_AREA)
  img_int_g = cv2.rotate(img_int_g, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_int_g.png", img_int_g)
  
  img_int_b = cv2.resize(img_in[3], dim, interpolation = cv2.INTER_AREA)
  img_int_b = cv2.rotate(img_int_b, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_int_b.png", img_int_b)
  
  img_DoLP_r = cv2.resize(img_in[4], dim, interpolation = cv2.INTER_AREA)
  img_DoLP_r = cv2.rotate(img_DoLP_r, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_DoLP_r.png", img_DoLP_r)
  
  img_DoLP_g = cv2.resize(img_in[5], dim, interpolation = cv2.INTER_AREA)
  img_DoLP_g = cv2.rotate(img_DoLP_g, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_DoLP_g.png", img_DoLP_g)
  
  img_DoLP_b = cv2.resize(img_in[6], dim, interpolation = cv2.INTER_AREA)
  img_DoLP_b = cv2.rotate(img_DoLP_b, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_DoLP_b.png", img_DoLP_b)
  
  img_AoLP_r = cv2.resize(img_in[7], dim, interpolation = cv2.INTER_AREA)
  img_AoLP_r = cv2.rotate(img_AoLP_r, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_AoLP_r.png", img_AoLP_r)
  
  img_AoLP_g = cv2.resize(img_in[8], dim, interpolation = cv2.INTER_AREA)
  img_AoLP_g = cv2.rotate(img_AoLP_g, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_AoLP_g.png", img_AoLP_g)
  
  img_AoLP_b = cv2.resize(img_in[9], dim, interpolation = cv2.INTER_AREA)
  img_AoLP_b = cv2.rotate(img_AoLP_b, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_AoLP_b.png", img_AoLP_b)
  
  img_water_r = cv2.resize(img_in[10], dim, interpolation = cv2.INTER_AREA)
  img_water_r = cv2.rotate(img_water_r, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_water_r.png", img_water_r)
  
  img_water_g = cv2.resize(img_in[11], dim, interpolation = cv2.INTER_AREA)
  img_water_g = cv2.rotate(img_water_g, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_water_g.png", img_water_g)
  
  img_water_b = cv2.resize(img_in[12], dim, interpolation = cv2.INTER_AREA)
  img_water_b = cv2.rotate(img_water_b, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite(name + "_water_b.png", img_water_b)
  
  
  
# Read images
img_name = sys.argv[1]
img_name_split = img_name.split("_")
if len(img_name_split) > 2:
  img_time = pd.to_datetime(img_name_split[2], utc=True)

img_raw = cv2.imread(img_name, 0)

if exists("flightinfo.csv"):
    DJI_data = pd.read_csv("flightinfo.csv", index_col=0)
    DJI_data["Image:Time"] = pd.to_datetime(DJI_data["Image:Time"], utc=True)

    for i in range(len(DJI_data["Image:Time"])):
      if 'img_time' in locals():
        if img_time == DJI_data["Image:Time"][i]:
          img_params = DJI_data.iloc[i]
          print(img_name)
          print(img_params)
          print()
    

img_polarsplit = pa.demosaicing(img_raw, "COLOR_PolarRGB")

#cv2.imwrite("0.bmp", img_polarsplit[..., 0])
#cv2.imwrite("45.bmp", img_polarsplit[..., 1])
#cv2.imwrite("90.bmp", img_polarsplit[..., 2])
#cv2.imwrite("135.bmp", img_polarsplit[..., 3])

#Calculate Stokes vectors
angles = np.deg2rad([90, 135, 0, 45])
img_stokes = pa.calcStokes(img_polarsplit, angles)

img_stokes_b = img_stokes[:,:,2,:]
img_stokes_g = img_stokes[:,:,1,:]
img_stokes_r = img_stokes[:,:,0,:]

img_mask = cv2.imread("assets/mask.png", 0)

img_intensity = pa.cvtStokesToIntensity(img_stokes).astype('float32')
img_intensity = cv2.cvtColor(img_intensity, cv2.COLOR_BGR2RGB)

img_intensity_r = pa.cvtStokesToIntensity(img_stokes_r)
img_intensity_g = pa.cvtStokesToIntensity(img_stokes_g)
img_intensity_b = pa.cvtStokesToIntensity(img_stokes_b)
    
overexposed_threshold = 230.0
img_overexposed_r = (img_intensity_r >= overexposed_threshold)
img_overexposed_g = (img_intensity_g >= overexposed_threshold)
img_overexposed_b = (img_intensity_b >= overexposed_threshold)
    
underexposed_threshold = 5.0
img_underexposed_r = (img_intensity_r <= underexposed_threshold)
img_underexposed_g = (img_intensity_g <= underexposed_threshold)
img_underexposed_b = (img_intensity_b <= underexposed_threshold)

img_intensity_r = cv2.cvtColor(img_intensity_r.astype('float32'), cv2.COLOR_GRAY2RGB)
img_intensity_r[:,:,0] = np.zeros([img_intensity_r.shape[0], img_intensity_r.shape[1]])
img_intensity_r[:,:,1] = np.zeros([img_intensity_r.shape[0], img_intensity_r.shape[1]])

img_intensity_g = cv2.cvtColor(img_intensity_g.astype('float32'), cv2.COLOR_GRAY2RGB)
img_intensity_g[:,:,0] = np.zeros([img_intensity_g.shape[0], img_intensity_g.shape[1]])
img_intensity_g[:,:,2] = np.zeros([img_intensity_g.shape[0], img_intensity_g.shape[1]])

img_intensity_b = cv2.cvtColor(img_intensity_b.astype('float32'), cv2.COLOR_GRAY2RGB)
img_intensity_b[:,:,1] = np.zeros([img_intensity_b.shape[0], img_intensity_b.shape[1]])
img_intensity_b[:,:,2] = np.zeros([img_intensity_b.shape[0], img_intensity_b.shape[1]])

img_intensity_r = cv2.bitwise_and(img_intensity_r, img_intensity_r, mask = img_mask)
img_intensity_g = cv2.bitwise_and(img_intensity_g, img_intensity_g, mask = img_mask)
img_intensity_b = cv2.bitwise_and(img_intensity_b, img_intensity_b, mask = img_mask)

img_DoLP_r = cvtStokesToDoLP(img_stokes_r)
img_DoLP_g = cvtStokesToDoLP(img_stokes_g)
img_DoLP_b = cvtStokesToDoLP(img_stokes_b)

img_AoLP_r = pa.cvtStokesToAoLP(img_stokes_r)
img_AoLP_g = pa.cvtStokesToAoLP(img_stokes_g)
img_AoLP_b = pa.cvtStokesToAoLP(img_stokes_b)

img_water_r = detectWater(img_intensity, img_DoLP_r, img_AoLP_r, img_overexposed_r, img_underexposed_r)
img_water_g = detectWater(img_intensity, img_DoLP_g, img_AoLP_g, img_overexposed_g, img_underexposed_g)
img_water_b = detectWater(img_intensity, img_DoLP_b, img_AoLP_b, img_overexposed_b, img_underexposed_b)

img_water_r = cv2.bitwise_and(img_water_r, img_water_r, mask = img_mask)
img_water_g = cv2.bitwise_and(img_water_g, img_water_g, mask = img_mask)
img_water_b = cv2.bitwise_and(img_water_b, img_water_b, mask = img_mask)

img_DoLP_r = applyColorToDoLP(img_DoLP_r, img_overexposed_r, img_underexposed_r)
img_DoLP_g = applyColorToDoLP(img_DoLP_g, img_overexposed_g, img_underexposed_g)
img_DoLP_b = applyColorToDoLP(img_DoLP_b, img_overexposed_b, img_underexposed_b)

img_DoLP_r = cv2.bitwise_and(img_DoLP_r, img_DoLP_r, mask = img_mask)
img_DoLP_g = cv2.bitwise_and(img_DoLP_g, img_DoLP_g, mask = img_mask)
img_DoLP_b = cv2.bitwise_and(img_DoLP_b, img_DoLP_b, mask = img_mask)

img_AoLP_r = applyAlgonetColorToAoLP(img_AoLP_r, img_overexposed_r, img_underexposed_r)
img_AoLP_g = applyAlgonetColorToAoLP(img_AoLP_g, img_overexposed_g, img_underexposed_g)
img_AoLP_b = applyAlgonetColorToAoLP(img_AoLP_b, img_overexposed_b, img_underexposed_b)

img_AoLP_r = cv2.bitwise_and(img_AoLP_r, img_AoLP_r, mask = img_mask)
img_AoLP_g = cv2.bitwise_and(img_AoLP_g, img_AoLP_g, mask = img_mask)
img_AoLP_b = cv2.bitwise_and(img_AoLP_b, img_AoLP_b, mask = img_mask)

#imwrite_result([img_intensity, img_intensity_r, img_intensity_g, img_intensity_b, img_DoLP_r, img_DoLP_g, img_DoLP_b, img_AoLP_r, img_AoLP_g, img_AoLP_b], img_name)
imwrite_result_split([img_intensity, img_intensity_r, img_intensity_g, img_intensity_b, img_DoLP_r, img_DoLP_g, img_DoLP_b, img_AoLP_r, img_AoLP_g, img_AoLP_b, img_water_r, img_water_g, img_water_b], img_name)

Figure(2482, 4268,
        Image(594, 709, img_name[:-4] + '_int.png').move(954, 177),
        Image(594, 709, img_name[:-4] + '_int_r.png').move(272, 1028),
        Image(594, 709, img_name[:-4] + '_int_g.png').move(954, 1028),
        Image(594, 709, img_name[:-4] + '_int_b.png').move(1637, 1028),
        Image(594, 709, img_name[:-4] + '_DoLP_r.png').move(272, 1782),
        Image(594, 709, img_name[:-4] + '_DoLP_g.png').move(954, 1782),
        Image(594, 709, img_name[:-4] + '_DoLP_b.png').move(1637, 1782),
        Image(594, 709, img_name[:-4] + '_AoLP_r.png').move(272, 2541),
        Image(594, 709, img_name[:-4] + '_AoLP_g.png').move(954, 2541),
        Image(594, 709, img_name[:-4] + '_AoLP_b.png').move(1637, 2541),
        Image(594, 709, img_name[:-4] + '_water_r.png').move(272, 3300),
        Image(594, 709, img_name[:-4] + '_water_g.png').move(954, 3300),
        Image(594, 709, img_name[:-4] + '_water_b.png').move(1637, 3300),
        Image(438, 68, 'assets/grey_bar.png').move(220, 668),
        Image(261, 256, 'assets/color_wheel.png').move(1778, 288),
        SVG('assets/DronePolarimetry_mask.svg')
       ).save(img_name[:-4] + '.svg')

f = open(img_name[:-4] + '.svg', 'r')
svg = f.read()
f.close()

scour_options = scour.sanitizeOptions(options=None) # get a clean scour options object
scour_options.remove_descriptive_elements = True # change any option you like
clean_svg = scour.scourString(svg, options = scour_options) # use scour

output_file = open(img_name[:-4] + '.svg', 'w')
output_file.write(clean_svg)
output_file.close()

os.remove(img_name[:-4] + '_int.png')
os.remove(img_name[:-4] + '_int_r.png')
os.remove(img_name[:-4] + '_int_g.png')
os.remove(img_name[:-4] + '_int_b.png')
os.remove(img_name[:-4] + '_DoLP_r.png')
os.remove(img_name[:-4] + '_DoLP_g.png')
os.remove(img_name[:-4] + '_DoLP_b.png')
os.remove(img_name[:-4] + '_AoLP_r.png')
os.remove(img_name[:-4] + '_AoLP_g.png')
os.remove(img_name[:-4] + '_AoLP_b.png')
os.remove(img_name[:-4] + '_water_r.png')
os.remove(img_name[:-4] + '_water_g.png')
os.remove(img_name[:-4] + '_water_b.png')