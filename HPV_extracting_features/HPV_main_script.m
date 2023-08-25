%% this script is used to show the example of how to use FeDeG (FLocK) MATLAB code to extract clusters index and features on MC

%% Intro
clear; clc; close all;


%% Making main folder and adding the data
main_path = HPV_making_path('DataFolder');

disp('Please add your input data (from Python code) in HPV_DataFolder (img and charac_lists) in the corresponding folder.');
disp('Then resume execution by typing "dbcont" in the command window')

keyboard;

%% Enter all the paths you want to use
% Input data
%img
used_image_for_plotting_MC_segmentation_path= 'modified_ouput_CALC1.png';
used_image_for_plotting_clusters_path ='input_00001.png';
%charac_lists
properties_path ='features_list.txt';
MC_locations_path ='MC_locations_list.txt';

% Output path to stock the data
%index
cluster_index_table_path='CALC1_estimated_cluster';
%centroids
cluster_centroids_table_path='TELIO_third';
%features
cluster_features_table_path='TELIO_first';
cluster_features_table_filename='444';
cluster_features_table_dataset='INbreast';
%exporting
csv_table_to_export_path='CALC1_estimated_cluster';
csv_table_exported_path='CALC1_estimated_cluster_exported';

%% Path
warning('off','all')
curpath=pwd;
addpath(genpath([curpath '/nuclei_seg']));

%% Reading the image for plotting MC segmentation
I = imread([pwd '\HPV_DataFolder\input\img\' used_image_for_plotting_MC_segmentation_path]);
I_normRed=I;

figure; imshow(I);

%% **** 1 MC segmentation 

p.scales=3:2:10;% the scale of MC

properties = HPV_load_properties_of_MC([pwd '/HPV_DataFolder/input/charac_lists/' properties_path]);
MC_locations = HPV_load_MC([pwd '/HPV_DataFolder/input/charac_lists/' MC_locations_path] );


show(I,1);hold on;

for k = 1:length(MC_locations)
    plot(MC_locations{k}(:,2), MC_locations{k}(:,1), 'g-', 'LineWidth', 2);
end
hold off;


%% Reading the image for plotting the clusters
I = imread([pwd '\HPV_DataFolder\input\img\' used_image_for_plotting_clusters_path]);
I_normRed=I;

%% **** 2 construct and visulize FeDeG 
% computing FeDeG using meanshift clustering
disp('use MS to build FeDeG...');

%%% feature space you would like to explore, cgeck function
%%% Lconstruct_FeDeG_v2 for more information, or you can add any feature
%%% combination there as well. Note that you should specify the bandWidth
%%% for spacial space and feature spaces, respectively (set bandwith in para.bandWidth_space, and para.bandWidth_features). 
para.feature_space='Centroid-Area-MeanIntensity'; %para.feature_space='Centroid-MeanIntensity';
para.bandWidth_space=80;% bandwidth in the spacial space, higher of the value, bigger of the FeDeG
para.bandWidth_features=[40;15];% bandwidth in the corresponding feature space

para.debug=1; % turn the debug mode on
para.nuclei=MC_locations; % assign pre-segmented MC_locations
para.properties=properties; % you can check out the nuclear properites 
para.shownuclei_thiness=1; % the visual effect of nuclei 
para.shownucleiconnection_thiness=3; % the visual effect of FeDeG

para.num_fixed_types=0; % specify the phenotype numbers in the image
[clustCent,data2cluster,cluster2dataCell,data_other_attribute,clust2types,typeCent]=Lconstruct_FeDeG_v3(I,para);

%% **** 3 extracting FeDeG features
para.I=I;
para.data_other_attribute=data_other_attribute;
para.debug=1;
para.clust2types=clust2types;
para.typeCent=typeCent;
[set_feature,set_feature_name]=L_get_FeDeG_features_v3(clustCent,cluster2dataCell,para);
set_feature_name=set_feature_name';



%% Saving
% Saving cluster index (like which points belong to which clusters) which could be used in python code later
HPV_extract_cluster_index(cluster_index_table_path,data2cluster,MC_locations,main_path)

% Saving the centroids of each clusters
HPV_extract_cluster_centers(cluster_centroids_table_path,clustCent,main_path)

% Saving cluster features (like cluster overlap etc...)
HPV_extract_cluster_features(cluster_features_table_path,cluster_features_table_filename,cluster_features_table_dataset,set_feature,set_feature_name,main_path)


%% Exporting one table from MATLAB csv format to a more general csv format (problems of delimiters)
HPV_exporting_matlab_csv_tables(csv_table_to_export_path,csv_table_exported_path,main_path)

