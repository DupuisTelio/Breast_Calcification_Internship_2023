import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import *
import cv2
import scipy
import time
import csv

#Trying new way of clusterisation
import sklearn
from scipy.spatial import distance
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation, Birch, MiniBatchKMeans, AgglomerativeClustering, DBSCAN

#Characteristics list for HPV
import skimage
import pickle

def telio():
    print("Hello world !\n")

#####################################################################################
"Binarisation and pre-treatment fonctions"
#####################################################################################


#####################################################################################
"Saving characateristics, locations and images functions"
#####################################################################################


#####################################################################################
"Loading cluster and FeDeG statistics from HPV (Matlab code)"
#####################################################################################


#####################################################################################
"Performing python clusterisation"
#####################################################################################


#####################################################################################
"Calculating metrics (AMD, Mc_IN,...)"
#####################################################################################
