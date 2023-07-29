%% this script is used to show the examplar of FeDeG (FLocK) using Lung TMA as exmaple

%% **** 0 read image
warning('off','all')
curpath=pwd;
addpath(genpath([curpath '/nuclei_seg']));

im_choice=2;

if im_choice==0
    % Nuclei image     %% To be able to compare nuclei clustering and microcalcification clustering
    I = imread([pwd '/img/TMA 002-G6.png']);
    % crop image
    I=imcrop(I,round(round([623.5 1132.5 510 414])));
    [I_norm, ~, ~] = normalizeStaining(I);
    I_normRed=I_norm(:,:,1);
elseif im_choice==1
    % plot only on microcalcification
    I = imread([pwd '\TELIO_img_for_HPV\modified_ouput2_CALC1.png']);
    I_normRed=I;
elseif im_choice==2
    % direclty on the mammogram image
    I = imread([pwd '\TELIO_img_for_HPV\input_00001.png']);
    I_normRed=I;
end


% load nuclei and properties  
load('param_.mat');
    

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
