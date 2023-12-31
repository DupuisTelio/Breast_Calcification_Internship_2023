%% this script is used to show the examplar of FeDeG (FLocK) using Lung TMA as exmaple

%% Intro
clear, clc


%% **** 0 read image
warning('off','all')
curpath=pwd;
addpath(genpath([curpath '/nuclei_seg']));

%% Initial example
%I = imread([pwd '/img/TMA 002-G6.png']);
% crop image
%I=imcrop(I,round(round([623.5 1132.5 510 414])));
%[I_norm, ~, ~] = normalizeStaining(I);
%I_normRed=I_norm(:,:,1);

% image already croped
%I = imread([pwd '/img/cropped.png']);
%[I_norm, ~, ~] = normalizeStaining(I);
%I_normRed=I_norm(:,:,1);

% image ng 0 255
%I = imread([pwd '/img/croppedng.png']);
%I_normRed=I;

% image mammo
I = imread([pwd '\TELIO_img_for_HPV_\modified_ouput_CALC1.png']);
I_normRed=I;

figure; imshow(I);

%% **** 1 neclei segmentation using multi-resolution watershed 

p.scales=3:2:10;% the scale of nuclei

%[nuclei, properties] = nucleiSegmentationV2(I_normRed,p);
properties = HPV_load_properties_of_MC('HPV_DataFolder/input/charac_lists/features_list.txt');
nuclei = HPV_load_MC('HPV_DataFolder/input/charac_lists//MC_locations_list.txt');


show(I,1);hold on;

for k = 1:length(nuclei)
    plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 2);
end
hold off;

%% Image_choice for plotting the clusters
im_choice=1;
if im_choice==0
    % plot only on microcalcification
    I = imread([pwd '\TELIO_img_for_HPV_\modified_ouput_CALC1.png']);
else
    % direclty on the mammogram image
    I = imread([pwd '\TELIO_img_for_HPV_\input_00001.png']);
end
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
para.nuclei=nuclei; % assign pre-segmented nuclei
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
