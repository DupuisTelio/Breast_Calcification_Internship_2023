import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import *
import scipy
import time


#Trying new way of clusterisation
import sklearn
from scipy.spatial import distance
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation, Birch, MiniBatchKMeans, AgglomerativeClustering, DBSCAN

#Characteristics and MC_locations list for HPV
import skimage
import pickle
import csv
import cv2

#Dicoms dataset
import pydicom
from PIL import Image



#####################################################################################
"Plotting functions"
#####################################################################################
def checking_loaded_mamm(processed_mamm):
    #check
    print("Image size:",processed_mamm.shape)
    print("Image type:",processed_mamm.dtype)

    #plot
    plt.figure()
    plt.imshow(processed_mamm,cmap = plt.cm.gray)

def checking_prediction(prediction):
    #check
    print("Image size:",prediction.shape)
    print("Image type:",prediction.dtype)

    # plot
    plt.figure()
    plt.imshow(prediction,cmap = plt.cm.gray)

def checking_everything_until_labels(processed_mamm,prediction,binary_image,label_image):
    plt.close('all')
    fig = plt.figure(figsize=(16,10))
    (ax1, ax2), (ax3,ax4) = fig.subplots(2, 2)

    ax1.imshow(processed_mamm,cmap = plt.cm.gray)
    ax1.set_title('Initial mammogram ')
    ax1.set_axis_off()

    ax2.imshow(prediction,cmap = plt.cm.gray)
    ax2.set_title('prediction ')
    ax2.set_axis_off()

    ax3.imshow(binary_image,cmap = plt.cm.gray)
    ax3.set_title('binary prediction  ')
    ax3.set_axis_off()

    nc = label_image.max()
    print("number of MC:", nc)

    ax4.imshow(label_image,cmap = plt.cm.gray)
    ax4.set_title('label image')
    ax4.set_axis_off()


def plotting_Mc_treated_for_HPV(processed_mamm,label_image):
    # prepare plot all MC
    fig = plt.figure()
    plt.suptitle('all MC until 25')
    axs = fig.subplots(5, 5)
    for k in range(25) :
        axs.flat[k].set_axis_off()

    props = skimage.measure.regionprops(label_image,processed_mamm)
    for region in props: #loop on MC 
        print("\n ==== MC or region label : ", region.label)
        #plot
        if region.label < 25 :
            regionNG = region.image_intensity
            regionMask = region.image_filled     #region.image
            pixelValues = regionNG * regionMask
            axs.flat[region.label].imshow(pixelValues,cmap = plt.cm.gray)




def plotting_different_python_clusterisation(clustering_choice,clustering_labels,regions,processed_mamm,binary_image):
    plt.imshow(processed_mamm)
    plt.scatter(regions[:, 1], regions[:, 0], c=labels_agglo, cmap='hot', s=1)

    # Invert axes to adapt to the mammogram plot
    plt.gca().invert_yaxis()
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title(f'{clustering_choice} : {n_clusters} clusters')
    plt.show()

    ## Based Image to compare with
    plt.imshow(processed_mamm)
    index_points = np.argwhere(binary_image == 1)
    plt.scatter(index_points[:, 1], index_points[:, 0], color='red', marker='o', s=1)

    # Invert axes to adapt to the mammogram plot
    plt.gca().invert_yaxis()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Superposition of calcifications on the mammogram')
    plt.show()




#####################################################################################
"Binarisation and pre-treatment fonctions"
#####################################################################################
def Histogram_and_choosing_threshold_value(prediction):
    # Chosing bins value
    N = len(prediction)

    #Square root rule
    #bins = int(np.ceil(np.sqrt(N)))

    #Sturges rule
    bins = int(np.ceil(np.log2(N) + 1))

    # Basic
    #bins=50

    print("Threshold value:",bins)

    # Ploting hist
    plt.hist(prediction, bins=bins)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of values with bins={bins}')
    return bins




def pre_treatment(prediction,threshold_value,fill_holes):
    # Copy
    binary_image = np.copy(prediction)

    # Binarization or Tresholding
    binary_image[binary_image<threshold_value] = 0
    binary_image[binary_image>= threshold_value] = 1
    if fill_holes:
        #eventual fill holes
        binary_image = scipy.ndimage.binary_fill_holes(binary_image) * 1

    #Mask
    prediction_modified = binary_image * prediction

    #Inversion (for HPV)
    prediction_modified = 1 - prediction_modified

    return prediction_modified,binary_image






#####################################################################################
"Calculating characteristics"
#####################################################################################
def labeling(binary_image):
    return skimage.measure.label(binary_image)



class Features :
    "microcalcifications features"
    def __init__(self):
        self.Centroid = 0
        self.Area = 0
        self.Eccentricity = 0
        self.Solidity = 0
        self.Circularity = 0
        self.MajorAxisLength = 0
        self.MinorAxisLength = 0
        self.MeanIntensity= 0

    def print_features(self):
        print('Centroid (%.2f %.2f)' % (self.Centroid))
        print('Area %d' % (self.Area))
        print('Eccentricicy %.2f' % (self.Eccentricity))
        print('Solidity %.2f' % (self.Solidity))
        print('Circularity %.2f' % (self.Circularity))
        print('MajorAxisLength %.2f' % (self.MajorAxisLength ))
        print('MinorAxisLength %.2f' % (self.MinorAxisLength ))
        print('MeanIntensity %.2f' % (self.MeanIntensity))


def Calculating_characteristics_and_MC_locations_for_HPV(processed_mamm,label_image):
    features_list = []
    features_list_for_HPV = []
    MC_locations_for_HPV = []

    props = skimage.measure.regionprops(label_image,processed_mamm)

    for region in props: #loop on MC to construct features_list

        #print("\n ==== MC or region label : ", region.label)

        # if the MC is too small we remove it
        if region.area < 2 or region.perimeter == 0 :
            a=1
            print(" too small ! area : %d ", region.area)
        else :

            fea = Features()

            # fill attributs directly python
            fea.Area =  region.area
            fea.Centroid = region.centroid
            fea.Eccentricity = region.eccentricity

            fea.Solidity = region.solidity
            fea.Circularity =  4 * math.pi * region.area / (region.perimeter**2)
            fea.MajorAxisLength = region.axis_major_length
            fea.MinorAxisLength = region.axis_minor_length

            # fill attributs mammo
            fea.MeanIntensity= region.intensity_mean

            MC_mam = region.image_intensity
            MC_Mask = region.image_filled     #region.image
            pixelValues = MC_mam * MC_Mask

            #fea.MeanIntensity= pixelValues.mean()
            fea.IntensityRange= np.percentile(pixelValues,97.5) - np.percentile(pixelValues,2.5)

            # matlab adaptation
            x = fea.Centroid[0]
            y = fea.Centroid[1]
            fea.Centroid = (y+1 , x+1)

            #fea.print_features()

            features_list.append(fea)

            features_list_for_HPV.append(y+1)
            features_list_for_HPV.append(x+1)
            features_list_for_HPV.append(fea.Area)
            features_list_for_HPV.append(fea.Eccentricity)
            features_list_for_HPV.append(fea.Solidity)
            features_list_for_HPV.append(fea.Circularity)
            features_list_for_HPV.append(fea.MajorAxisLength)
            features_list_for_HPV.append(fea.MinorAxisLength)
            features_list_for_HPV.append(fea.MeanIntensity)

            temporary=[]
            for coord in region.coords:
                temporary.extend(coord+1)
            #MC_locations.append(round(len(temporary)/2)) # the number of pixels is added in head
            MC_locations_for_HPV.extend(temporary) # the pixels are added
            MC_locations_for_HPV.append(0) # to separate each region

    print("\n Number of MC kept for their characteristics (not too small): %d " ,len(features_list))
    print("\n Number of MC kept for their characteristics (not too small): %d " ,len(features_list_for_HPV)/9)
    MC_locations_for_HPV.insert(0,len(features_list)) # number of MC kept

    return MC_locations_for_HPV,features_list_for_HPV,features_list



#####################################################################################
"Saving characteristics, locations and images functions"
#####################################################################################
def Saving_characteristics_HPV_format(directory_path_characteristics,mammog_name,MC_locations,features_list,chosen_format="txt"):
    #chosen_format: txt or csv but txt recommanded for HPV
    if not os.path.exists(directory_path_characteristics):
        os.makedirs(directory_path_characteristics)

    file_path_features = os.path.join(directory_path_characteristics, f'{mammog_name}_features_list_for_HPV.{chosen_format}')
    file_path_MClocations = os.path.join(directory_path_characteristics, f'{mammog_name}_MC_locations_list_for_HPV.{chosen_format}')

    if chosen_format =="txt":
        # Save the list of features to the text file
        with open(file_path_features, 'w') as file:
            for num in features_list:
                file.write(str(num) + '\n')
        # Save the list of MC_locations to the text file
        with open(file_path_MClocations, 'w') as file:
            for num in MC_locations:
                file.write(str(num) + '\n')

    elif chosen_format =="csv":
        # Save the list of features to the CSV file
        with open(file_path_features, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for num in features_list:
                csv_writer.writerow([num])
        # Save the list of MC_locations to the CSV file
        with open(file_path_MClocations, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for num in MC_locations:
                csv_writer.writerow([num])


def Saving_images_used_and_produced(directory_path_img,mammog_name,processed_mamm,prediction,binary_image,chosen_format="png"):
    #chosen_format: png or tif
    if not os.path.exists(directory_path_img):
        os.makedirs(directory_path_img)
    # Loaded mammog
    file_path1 = os.path.join(directory_path_img, f'{mammog_name}_processed_mamm.{chosen_format}')
    # First prediction
    file_path2 = os.path.join(directory_path_img, f'{mammog_name}_first_prediction.{chosen_format}')
    # Modified prediction
    file_path3 = os.path.join(directory_path_img, f'{mammog_name}_binary_image.{chosen_format}')
  
    if chosen_format =="png": 
        processed_mamm = processed_mamm * 255
        prediction = prediction * 255
        binary_image = binary_image * 255

    # Saving
    cv2.imwrite(file_path1, processed_mamm)
    cv2.imwrite(file_path2, prediction)
    cv2.imwrite(file_path3, binary_image)



#####################################################################################
"Loading cluster and FeDeG statistics from HPV (Matlab code)"
#####################################################################################





#####################################################################################
"Performing python clusterisation"
#####################################################################################
def Trying_python_clusterisation(clustering_choice,binary_image,n_clusters=5,epsilon=10,min_samples=4):
    # Locations of calcifications
    regions = np.argwhere(binary_image == 1)

    # Matrix of distance between regions
    dist_matrix = distance.cdist(regions, regions, metric='euclidean')

    t0=time.time()

    if clustering_choice==AgglomerativeClustering:
        n_clusters = 20
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
    if clustering_choice==DBSCAN:
        # epsilon: maximum distance between two samples for them to be considered as part of the same neighborhood (image size)
        # min samples: minimum number of samples in a neighborhood for a point to be considered a core point (density of points)
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
    if clustering_choice==KMeans:
        clustering = KMeans(n_clusters=n_clusters)
    if clustering_choice==MeanShift:
        clustering = MeanShift()
    if clustering_choice==SpectralClustering:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    if clustering_choice==AffinityPropagation:
        clustering = AffinityPropagation(affinity='precomputed')
    if clustering_choice==Birch:
        clustering = Birch(n_clusters=n_clusters)
    if clustering_choice==MiniBatchKMeans:
        clustering = MiniBatchKMeans(n_clusters=n_clusters)

    clustering_labels= clustering.fit_predict(dist_matrix)
    print(f"{clustering_choice}, Time spent:{time.time()-t0}")
    return clustering_labels,regions








#####################################################################################
"Calculating metrics (AMD, Mc_IN,...)"
#####################################################################################








#####################################################################################
"Dicoms dataset read and transformation to png"
#####################################################################################
def transforming_Dicoms_to_png(input_folder,output_folder):
    #WARNING you might need to do : !pip install pydicom

    os.makedirs(output_folder, exist_ok=True) #making the output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        if filename.endswith('.dcm'):
            print(f"Traitement du fichier {filename}")  # Print the name of the file treated 
            try:
                # Load Dicoms file
                path = os.path.join(input_folder, filename)
                ds = pydicom.dcmread(path)

                # Convert Dicoms to PIL Image
                im = Image.fromarray(ds.pixel_array)

                # Convert the image to 8 bits
                image_16bit = np.array(im)
                image_8bit = ((image_16bit - image_16bit.min()) / (image_16bit.ptp() / 255.0)).astype(np.uint8)
                image_8bit = 255 - image_8bit  # Inverse
                im_8bit = Image.fromarray(image_8bit)

                # Saved with JPG format
                new_filename = os.path.splitext(filename)[0] + '.png'  # Add '.jpg' extension at the file name
                im_8bit.save(os.path.join(output_folder, new_filename), "PNG")
                print(f"Image saved : {new_filename}")  
            except Exception as e:
                print(f"Error in the treatment of {filename}: {e}")  # in case of error
        else:
            print(f"The file {filename} isn't .dcm")  