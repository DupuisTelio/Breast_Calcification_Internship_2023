%% this script is used to show the examplar of FeDeG (FLocK) using Lung TMA as exmaple

%% **** 0 read image
warning('off','all')
curpath=pwd;
addpath(genpath([curpath '/nuclei_seg']));

choice=1;

if choice==0
    % Nuclei image
    I = imread([pwd '/img/TMA 002-G6.png']);
    % crop image
    I=imcrop(I,round(round([623.5 1132.5 510 414])));
    [I_norm, ~, ~] = normalizeStaining(I);
    I_normRed=I_norm(:,:,1);
else
    % mammo image
    I = imread([pwd '\TELIO_modified_ouput_ex.png']);
    I_normRed=I;
end
figure; imshow(I);

%% **** 1 neclei segmentation using multi-resolution watershed 

p.scales=3:2:10;% the scale of nuclei

[nuclei, properties] = nucleiSegmentationV2(I_normRed,p);

show(I,1);hold on;

for k = 1:length(nuclei)
    plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 2);
end
hold off;

%% Saving
save('param_.mat', 'nuclei', 'properties');
    
