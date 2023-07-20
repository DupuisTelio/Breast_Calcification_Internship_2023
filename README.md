# BreathCalcificationSummer2023
Here is the repository containing the code and diapositives I worked on during my internship in HMU-ISCA LAB (Greece) from June to August 2023

This project focus on microcalcification detection in mammograms using deep learning models, and features extraction from those detected microcalcifications. 

** Calc1 folder : Containing the code about microcalcifications detection, the starting code and the modifications apart from it

** Slides folder : Containing all the slides used at the meetings, providing a clear and compact summary of progress

** HPV folder : Containing the code about features extraction, the starting code (tailored for nuclei and not microcalcifications) and the modifications apart from it

** Publications_used folder : Containing all of the publications considered during this internship

In order to make the code run, you need to run "CalcificationDetectionTelio.ipynb" from Calc1 folder then "HPV_step1and2.m" from HPV folder on the output image of the jupyter code.



Here is a summary of my work during this internship:
---

To begin, my task is after making a little state of the art and understanding the problems involved, making a short presentation about it :

--16/06 presentation diapositive : https://tome.app/staaage/template-product-design-review-clixv8r1s0kwjpf3dv24xw0ma

New objectives:

Focus on microcalcification shaped like dots/clusters of dots: detecting them and being able to derive statistics such as the size of these clusters, their shape, the distance separating them, etc. 

In order to achieve this, we start from an existing code for the detection of microcalcifications in the form of dots ("Segmenting Microcalcifications in Mammograms and its Applications" which will be designed by "CALC1" later on) and we try to apply to it the study of distributions carried out in "Feature-driven Local Cell Graph (FLocK): New Computational Pathology-based Descriptors for Prognosis of Lung Cancer and HPV Status of Oropharyngeal Cancers" which will be designed by "HPV" later on.

Then if I succeed maybe, thanks to a radiologist's diagnoses, to see if it is possible to make a diagnosis based on these characteristics, or at least extracting information from those characteristics.

--- 
From 16/06 to 03/07:

We have spent most of our time studying the code in order to combine HPV and CALC1, specifically the HPV code, which consists of two steps: the first step performs cell nucleus segmentation using a watershed algorithm (morphology), and the second step performs clustering on the output of the first step using deep learning. 
We have particularly focused on the inputs and outputs of calc1, HPV step1, and step2.

Based on this, we have considered two solutions. The first solution is to directly use the output image of calc1 as the input for step1. However, this would require adapting the watershed algorithm since the second segmentation is not perfect as it is originally designed for microcalcifications and not cell nuclei. The second solution is to extract the necessary properties from the output image of calc1 ourselves, in order to use this information as input for step2. (more details in the diapositive of 03/07 presentation)

--03/07 presentation diapositive on the repository

New objectives proposed:

Firstly, superimpose the clusters detected using the first proposed approach onto the mammogram images.
Next, attempt to form clusters in a simple manner, such as by considering the distance separating the microcalcifications, without utilizing the HPV code.
Additionally, search for new publications regarding the clustering of microcalcifications.
Conduct a study on radiomic properties and establish my own list of properties for the second proposed approach, and modify the HPV code accordingly to accommodate this new list of properties.


--- 
From 03/07 to 17/07:
  Some new publications have been found and studied but none of great interest for our specific case
  We have superimposed the clusters detected using the first proposed approach onto the mammogram images and critized this first solution presented last week.
  Also a new simple way to form clusters based on distance and pre-fixed numbers of clusters has been implemented.
  Furthermore, progress has been on made on establishing an equivalent (in python) characteristics list for solution 2 (beginning directly with step 2 of HPV).

--17/07 presentation diapositive on the repository

New objectives proposed:

  Have a deeper look to CALC1 model to study the impact of the threshold for the prediction (lower -> more areas detected) and how they determine it for their specific dataset (try to tailor to our CardioCare Dataset)

  About the clustering: the numbers of clusters isn't really relevant. Characteristics, such as how pack the clusters are, the overlap, specific spatial characterisics, ... are the think that should be searched -> pursue with the FLOCK/FEDEG (HPV code)

  Try to find interesting metric about the clusters once detected : such as the mean of the minimal distance between them, the radius of the cluster and the number of other MC in this radius. Also a new publication has been suggested : https://www.nature.com/articles/s41467-023-37822-0 (or available on this Github ) that might bring solution for those characterics. With this publication, the focus should be only on the fig3 and the measures for only 1 class of cell. 

