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

from skimage import util, color, io # according to tutorial cv2 failed to do measure on image
from skimage import filters



# step 1: read collection of image and convert to gray, and float form

import glob#read data in a file


path = "C:/Users/Laga Ash/Desktop/FIB1/*.tif"
for file in glob.glob(path): 
    img = util.img_as_float(io.imread(file, 0))# 0 means read as gray
    pixel_to_um = 0.126 # 1 pixel = 0.126 um
    basename = PurePath(file).stem#extract file name
    
    #############################################################
    # step 2: denoising, and blur images use different filters in ndimage
    
    #1:check the histogram of the image to rescale image;
    #histogram need 1D array, image is 2D, so need to convert image into 1D
    #io.imshow(img)# check whether image need to denoise       
    #plt.hist(img.flat, bins = 100, range = (0, 0.5))
    
    #2: remove small particles in the image
    from skimage.morphology import disk, white_tophat
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
    
    #3: denoise try different filter
    gaussian_img = filters.gaussian(cleared_img, sigma = 0.5)
   
    median_img = filters.median(cleared_img, selem1, behavior = 'ndimage')
    """
    sigma: Standard deviation for Gaussian kernel. 
    The standard deviations of the Gaussian filter are given for each axis as a sequence, 
    or as a single number, in which case it is equal for all axes.
    smaller sigma preserve edge better: to differentiate the central pixel from surrounding pixels
    """
    #########################################################################
    # step 3: thresholding and clean up image(erode, fill holes, remove small objects etc) 
             #and create mask for image
       
    #1: thresholding to create a binary image
    #from skimage import filters
    thresh_gaussian_filter = filters.threshold_otsu(gaussian_img)# the value for doing threshold
    binary_img1 = gaussian_img > thresh_gaussian_filter
   
    ###############################################
    #edge detected segmentation
    #from skimage.filters import sobel      
    
    #find edges using the Sobel filter
    edges = filters.sobel(gaussian_img)
    
    #convert images into boolean array
    thresh_sobel_filter = filters.threshold_otsu(edges)
    binary_img2 = edges> thresh_sobel_filter
    
    # creating mask
    from skimage.morphology import erosion
    eroded_img = erosion(binary_img2, selem=disk(1))   
    filled_contours_img = ndi.binary_fill_holes(eroded_img).astype(int)#filling holes
    
    # watershed to distinguish two overlapping images
    #watershed
   
    from skimage.segmentation import watershed, clear_border
    from skimage.feature import peak_local_max
    from skimage.morphology import opening

    """
    ndi.distance_transform_edt():
        Exact Euclidean distance transform, which gives values of the Euclidean distance:
                           n
            y_i = sqrt(sum (x[i]-b[i])**2)
                          i
        where b[i] is the background point (value 0) with the smallest Euclidean distance to input points x[i], 
        and n is the number of dimensions.
        
    peak_ocal_max():
        Find peaks in an image as coordinate list or boolean mask.
        Peaks are the local maxima in a region of 2 * min_distance + 1 (i.e. peaks are separated by at least min_distance).
        If there are multiple local maxima with identical pixel intensities inside the region defined with min_distance, 
        the coordinates of all such pixels are returned.
   min_distance: int, optional
        Minimum number of pixels separating peaks in a region of 2 * min_distance + 1 (i.e. peaks are separated by at least 
        min_distance). To find the maximum number of peaks, use min_distance=1.
  markers: 
      int, or ndarray of int, same shape as image, optional
      The desired number of markers, or an array marking the basins with the values to be assigned in the label matrix. 
      Zero means not a marker. If None (no markers given), the local minima of the image are used as markers.
    indices:
        If indices = True : (row, column, …) coordinates of peaks.
        If indices = False : Boolean array shaped like image, with peaks represented by True values.
    footprint:
        ndarray of bools, optional
        If provided, footprint == 1 represents the local region within which to search for peaks at every point in image. 
        Overrides min_distance.
    labels: 
        ndarray of ints, optional
        If provided, each unique region labels == value represents a unique region to search for peaks. 
        Zero is reserved for background.
        An integer ndarray where each unique feature in input has a unique label in the returned array.
    mask:
        ndarray of bools or 0s and 1s, optional
        Array of same shape as image. Only points at which mask == True will be labeled.
    """
   
    distance = ndi.distance_transform_edt(filled_contours_img)#euclidean distrance transform between each pixels
    local_max = peak_local_max(distance, indices=False, footprint=np.ones((1, 1)))
    
    """
    ndi.label():
    Label features in an array. An array-like object to be labeled. 
    Any non-zero values in input are counted as features and zero values are considered the background.
    """
    markers = ndi.label(local_max)[0]
    watershed_img = watershed(distance, markers=markers, mask=filled_contours_img) 
    cleared_img = opening(watershed_img, selem=disk(2))
    
    #remove artifacts connected to image border
    final_mask = clear_border(cleared_img)
    
    #####################################################################
    
    # step 4: label image regions, label the object in the mask
    
    from skimage.measure import label, regionprops
    from skimage.color import label2rgb
    
    #1. label mask with different colors
    color_label_img = label(final_mask)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(color_label_img, bg_label=0)
    
    #2. label the mask with integer
    connectivity = [[1,1,1], [1,1,1], [1,1,1]] #8-connectivity, imageJ default
    # label_im, nb_labels = ndimage.label(mask), ndimage has a function to label unconnected pixels
    #from scipy import ndimage as ndi
    numerically_labeled_mask, num_labels = ndi.label(final_mask, structure=connectivity)
    #The function outputs a new image that contains a different integer label 
    #for each object, and also the number of objects found.
    #3. color the labels to see the effect
    nm_labeled_mask = color.label2rgb(numerically_labeled_mask, bg_label=0)
       
    ##########################################################
    # step 5. measure the properties of each labeled object
    props = regionprops(numerically_labeled_mask, img)
    
    ###################################################################
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
    titles = ['Original', 'gaussian blur', 'segmented', 'labeled']
    imgs = [img, gaussian_img,
            final_mask, nm_labeled_mask]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')
    plt.tight_layout()
    plt.savefig("C:/Users/Laga Ash/Desktop/FIB1/{0} {1}.jpg".format(basename, "layout"))
