# BreathCalcificationSummer2023
Here is the repository containing the code I worked on during my internship in HMU-ISCA LAB (Greece) from June to August 2023

This project focus on Calcification detection in mammograms using deep learning models. My task is after making a little state of the art and understanding the problems involved, making a short presentation about it :

--16/06 presentation diapositive : https://tome.app/staaage/template-product-design-review-clixv8r1s0kwjpf3dv24xw0ma

Then to focus on microcalcification shaped like dots/clusters of dots: detecting them and being able to derive statistics such as the size of these clusters, their shape, the distance separating them, etc. 

In order to achieve this, we start from an existing code for the detection of microcalcifications in the form of dots ("Segmenting Microcalcifications in Mammograms and its Applications" which will be designed by "CALC1" later on) and we try to apply to it the study of distributions carried out in "Feature-driven Local Cell Graph (FLocK): New Computational Pathology-based Descriptors for Prognosis of Lung Cancer and HPV Status of Oropharyngeal Cancers" which will be designed by "HPV" later on.

Then if I succeed maybe, thanks to a radiologist's diagnoses, to see if it is possible to make a diagnosis based on these characteristics, or at least extracting information from those characteristices.

---
We have spent most of our time studying the code in order to combine HPV and CALC1, specifically the HPV code, which consists of two steps: the first step performs cell nucleus segmentation using a watershed algorithm (morphology), and the second step performs clustering on the output of the first step using deep learning. 
We have particularly focused on the inputs and outputs of calc1, HPV step1, and step2.

Based on this, we have considered two solutions. The first solution is to directly use the output image of calc1 as the input for step1. However, this would require adapting the watershed algorithm since the second segmentation is not perfect as it is originally designed for microcalcifications and not cell nuclei. The second solution is to extract the necessary properties from the output image of calc1 ourselves, in order to use this information as input for step2.

--More details in the diapositive of 03/07 presentation

