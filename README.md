# BreastCalcificationSummer2023
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

--- 

From 17/07 to 28/07:

A deeper study of CALC1 has been made to prove that the output is definitely a probablity of MC and the use of a threshold is only made after for the plots (so adding a threshold is justified in our case to  have a binary output, and this threshold is a bit lower than the one they put as an init value for their plot so more MC area can be found).

FLOCK/FEDEG -> Solution is now completed, with functions in python that calculate the properties list then extract it in a txt file with a specific format, so that 2 matlab functions newly added can read thoses lists and put it as enter of step 2 (we avoid now step 1)
The results are good in terms of clusters making and statistics about those clusters. Now we need to choose on what MC properties we want to perform our clusters.

7 new ways of clustering tried in python but now that solution 2 is providing us a solid clustering, we will probably not pursue in this aspect.

New metrics have been found studying the SPIAT (the new publication from last meeting, R language ) code, one have been implemented : the AMD (Average Minimal Distance) for MC of all the images, MC inside a specific cluster, MC of all the images but only compared with others MC inside their own cluster.
Now we should try to find a way to take the clusters in matlab (output of HPV/FeDeG code) and put it as an entry of the AMD code in python. Also make this AMD between clusters and an other metric (found before the meeting) could be implemented about the number of MC in a given radius R.

Made a script to turn the dicoms dataset in png, but didn't try yet with this dataset

--28/07 presentation diapositive on the repository

New objectives proposed:

  Try with dicoms dataset, and ddsm dataset, IN breast dataset (clusters and ground truth) to compare ground truth, with the model
-> ddsm, IN breast dataset :https://forthgr-my.sharepoint.com/:p:/g/personal/stamoulou_ics_forth_gr/EU4_Whd_lbhNtbBXvdNKpQwBOCTt12mon0iAgw5qCCYEqg?e=wFbkxC

  As said above:
adapt matlab clusters output to put it in python and calculate AMD / other metrics 
End AMD ( between clusters), do radius and maybe try new metrics ? 

  Own settled objective:
Cleaning the python code because it’s starting to be really messy
Maybe start working on a script to make the whole process easier of access


  Create a way to extract MC locations (as I did in txt) in csv file for Georges and Elisaveth personnal work -> what kind of format would they prefer ?


 --- 

From 29/07 to 04/08:

2 news datasets (Cardiocare and INbreast) have been used to try the second solution on. Pretty convenient results, still need to compare with ground truth.

Most of the functionning jupyter code has been cleaned and put in 2 python file : new_core.py and auxiliar_functions.py accessible on this github.

New function made to save MC_locations and properties detected in a csv file, what kind of format would George and Elisaveth (my supervisors) prefer as an entry of their algorithms?

A deeper study of the parameters on which we perform the clustering, which are relevant or not?

2 new python metrics : AMD_between_clusters (border to border, 3 functions ) and McIN (Nb of Mc given a radius R) -> How do we choose R?

Advance on exctracting matlab output but not finished yet.

--04/08 presentation diapositive on the repository

New objectives proposed:

Change the functions to save MC_locations and properties in the specific format of INbreast, and respecting George code's input (row to row ): each row a MC, patient_ID, extra-information,...

For the AMD do centroid-centroid not borders-borders distance, it's easier even if maybe loosing a bit information, George already have it in his code, and I should keep borders-borders because even if it's more complicated it might interesant later on.

Have a look on the graphs of flock after clustering and before FeDeG stats
FeDeG have a look if needs clusters before FeDeG stats or not ? (probably yes)

Remove errors on borders in calc1 -> use general mask, look for a threshold ?

Find a way to evaluate the performance


 --- 

From 05/08 to 09/08:

Only 2 days (monday and tuesday 07-08/08) because of the weekend, so not so much advances there regarding the last objective settled.

New convincing tries on INbreast dataset.
Size problems while loading the tremendous DDSM dataset with their specific software (NBIA data retriever) on my non-powerfull laptop.

Cleaning the python code, a lot of jupyter functions have been add in the 2 (presented last time) python functions.

Problem reading Dicoms, converting to png =loosing information -> new approach working directly on the dicoms image
Changing some new_core.py functions in order to adapt to 3D images and dicoms image

Advance on adaptation of matlab outputs

(Finishing) Implementation of the saving function in the expected table format, overwriting option, folder,...

Problems with high-breast density mammogram (INbreast)


--09/08 presentation diapositive on the repository

New objectives proposed:

Smalls change on the saving function for the table ( Patient_id -> file_name, Add number of mc, add extra_information), send a table with 5 patients to George so that he can use it as an input of his code 
  
Estimate accuracy with ground truth (INbreast xml file)

Receive a part of DDSM dataset loaded by Grigorios (thanks) on a more powerfull machine

Breast density inBreast -> threshold ?


