# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:24:56 2020

@author: Laga Ash
"""


# -*- coding: utf-8 -*-
"""
This code performs image segmentation with skimage
Step 1: Read image and convert image dtype to easy to calculate
Step 2: Denoising, if required 
step 3: threshold image Clean up image, if needed (erode, etc.) 
        and create a mask for image
Step 4: Label grains in the masked image
Step 5: Measure the properties of each grain (object)
Step 6: Output results into a csv file

"""

# step O: import needed functions
import numpy as np #do calculation on images
from matplotlib import pyplot as plt
from matplotlib import axes#to show images in plot form
from pathlib import PurePath
from scipy import ndimage as ndi #the statistic libary in python to process n-dimensional image

from skimage import img_as_float, color, io # according to tutorial cv2 failed to do measure on image
from skimage import filters

import glob#read data in a file

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import disk, white_tophat, reconstruction
from skimage.morphology import erosion, dilation

from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max, canny

from skimage.measure import label, regionprops
from skimage.color import label2rgb

# step 1: read collection of image and convert to gray, and float form
"""
import glob#read data in a file
import os#extract file name
"""
path = "C:/Users/apeng/Desktop/PENG/FIB1/*.tif"
for file in glob.glob(path): 
    img = img_as_float(io.imread(file, 0))# 0 means read as gray
    pixel_to_um = 0.126 # 1 pixel = 0.126 um
    basename = PurePath(file).stem#extract file name
    # step 2: denoising, and blur images use different filters in ndimage
    
    #1:check the histogram of the image to rescale image;
    #histogram need 1D array, image is 2D, so need to convert image into 1D
    #io.imshow(img)# check whether image need to denoise       
    #plt.hist(img.flat, bins = 100, range = (0, 0.5))
    
    #2: remove small particles in the image
    #from skimage.morphology import disk, white_tophat
    """
    white_tophat: subtract morphological opening from image
                = image - opening
    opening = erosion + dilation: remove small particles
    return bright spots smaller than target elements
    """
    selem1 = disk(1)
    """disk(radius, dtype=<class 'numpy.uint8'>) is to generates a flat, disk-shaped structuring element.
    A pixel is within the neighborhood if the Euclidean distance between it and the origin is no greater than radius
    return selem: ndarray. The structuring element where elements of the neighborhood are 1 and 0 otherwise.
    """
    small_particles_img = white_tophat(img, selem1)
    cleared_img = (img - small_particles_img)
    
    #2: denoise try different filter
    gaussian_img = filters.gaussian(cleared_img, sigma = 0.5)
    print(gaussian_img.dtype)
    median_img = filters.median(cleared_img, selem1, behavior = 'ndimage')
    """
    sigma: Standard deviation for Gaussian kernel. 
    The standard deviations of the Gaussian filter are given for each axis as a sequence, 
    or as a single number, in which case it is equal for all axes.
    smaller sigma preserve edge better: to differentiate the central pixel from surrounding pixels
    """
    # step 3: thresholding and clean up image(erode, fill holes, remove small objects etc) 
             #and create mask for image
       
    #1: thresholding to create a binary image
    #from skimage import filters
    thresh = filters.threshold_otsu(gaussian_img)# the value for doing threshold
    
    binary = gaussian_img > thresh
   
    ###############################################
    #region based segmentation
    #from skimage.filters import sobel
            
    #show the histogram of image
    from skimage.exposure import histogram
    hist, hist_centers = histogram(gaussian_img)
    plt.plot(hist_centers, hist, lw = 1)
    
    #threshold based on histgram value
    plt.imshow(gaussian_img > 0.1, cmap = plt.cm.gray)    
    
    #find edges using the Sobel filter
    edges = filters.sobel(gaussian_img)
    plt.imshow(edges, cmap=plt.cm.gray)
    hist, hist_centers = histogram(edges)
    plt.plot(hist_centers, hist, lw=1)
    ##########################################filling holes
    #rescale image to stretch or shrink intensity levels, to clearly differentiate low intensity to high
    from skimage.exposure import rescale_intensity
    rescaled_edges = rescale_intensity(edges, in_range=(0, 255))
    plt.imshow(rescaled_edges)
    print(rescaled_edges)
   
    #find markers of the background and the image based on the extreme parts of the histogram of gray values.
    markers = np.zeros_like(gaussian_img)
    markers[gaussian_img < 0.01] = 1
    markers[gaussian_img > 0.1] = 2    

    #2: clean up image: fill holes, watershed to distinguish two overlapping images
    
    # 1): morphology of the mask
    """
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat
    from skimage.morphology import disk, reconstruction
    """
    seed = np.copy(edges)   
    seed[1:-1, 1:-1] = edges.max()
    mask = edges
    filled_holes = reconstruction(seed, mask, method = 'erosion')
    plt.imshow(filled_holes, cmap=plt.cm.gray)
    # 2): segragate two smoothingly connected patches
    #watershed
    """
    from skimage.segmentation import watershed, clear_border
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    """
    distance = ndi.distance_transform_edt(dilated)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((1, 1)))
    #print(local_maxi, local_maxi.dtype)
    markers = ndi.label(local_maxi)[0]
    #print(markers, markers.dtype)
    labels = watershed(distance,markers=markers, mask =dilated) 
    
    #remove artifacts connected to image border
    cleared = clear_border(labels)
    
    # step 4: label image regions, label the object in the mask
    """
    from skimage.measure import label, regionprops
    from skimage.color import label2rgb
    """
    #1. label mask with different colors
    label_img = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_img, bg_label=0)
    
    #2. label the mask with integer
    s = [[1,1,1], [1,1,1], [1,1,1]] #8-connectivity, imageJ default
    # label_im, nb_labels = ndimage.label(mask), ndimage has a function to label unconnected pixels
    #from scipy import ndimage as ndi
    labeled_mask, num_labels = ndi.label(cleared, structure=s)
    #The function outputs a new image that contains a different integer label 
    #for each object, and also the number of objects found.
    #3. color the labels to see the effect
    nm_labeled_mask = color.label2rgb(labeled_mask, bg_label=0)
    #print(num_labels)   
    
    # step 5. measure the properties of each labeled object
    props = regionprops(labeled_mask, img)
    """
    try to label each cluster with number but haven't figure out
    from PIL import ImageDraw
    import cv2
    output_img = nm_labeled_mask.copy()
    for i in range(1, num_labels + 1):
            print (i)       
    for properties in props:
        y0, x0 = properties.centroid
        x = int(x0)
        y = int(y0)
              
        labled = cv2.putText(output_img, text = "{}".format(i), org = (x, y),fontFace = 1, 
                                 fontScale = 2, color = (255, 255, 255), thickness = 2)
        plt.imshow(labled)
        plt.show()
    """
    # Step 6: Output results into a csv file
    #1. the property list
    propList = ['Area','MajorAxisLength','MinorAxisLength','Perimeter','Compactness', 
                'Eccentricity', 'circularity', 'Convexity','Aspect ratio','solidity',
                'MinIntensity', 'MeanIntensity', 'MaxIntensity']
    #2. creat an csv file to store properties
    output_file = open("C:/Users/Laga Ash/Desktop/FIB1/{0} {1}.csv".format(basename, "measurement"), "w")
    output_file.write(',' + ",".join(propList) + '\n') #join strings in array by commas, "," first comma is to leave first cell blank
    #3. to write properties in cells
    import math
    for cluster_props in props:      
        #output cluster properties to the excel file
        output_file.write(str(cluster_props['Label']))
        for i, prop in enumerate(propList):
            if(prop == 'Area'): 
                to_print = cluster_props[prop]*pixel_to_um**2   #Convert pixel square to um square
            elif(prop == 'Compactness'):               
                to_print = (4*cluster_props['area']*math.pi) /cluster_props['perimeter']**2
            elif(prop == 'circularity'):
                convex_img = cluster_props['convex_image'].astype('uint8')
                #print(convex_img.dtype)
                for props_convex in regionprops(convex_img):                          
                    to_print = (4*cluster_props['area']*math.pi) /props_convex['perimeter']**2
            elif(prop == 'Convexity'):
                to_print = props_convex['perimeter']/cluster_props['perimeter']
            elif(prop == 'Aspect ratio'):
                if cluster_props['major_axis_length'] == 0 or cluster_props['minor_axis_length'] == 0:
                    to_print = 'N/A'
                else:
                    to_print = cluster_props['major_axis_length'] /cluster_props['minor_axis_length']
            elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
                to_print = cluster_props[prop]*pixel_to_um
            else: 
                to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
            output_file.write(',' + str(to_print))
        output_file.write('\n')
    output_file.close()   #Closes the file, otherwise it would be read only. 
    
    #show images
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax = axes.ravel()
    titles = ['Original', 'gaussian blur', 'watershed', 'labeled']
    imgs = [img, gaussian_img,
            labels, nm_labeled_mask]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')
    plt.tight_layout()
    plt.savefig("C:/Users/Laga Ash/Desktop/FIB1/{0} {1}.jpg".format(basename, "layout"))
   
    
