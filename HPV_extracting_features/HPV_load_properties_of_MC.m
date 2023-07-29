function properties_MC = HPV_load_properties_of_MC(filename)


%% Load the data from the text file
data = load(filename); 


%% Create a struct array similar to the one of FeDeG code (properties array)

% Number of elements
n_elements = length(data)/9;

% Init the tab
properties_MC = repmat(struct(), n_elements, 1);

%% Main loop
for i = 1:n_elements
    index_shift=9*(i-1);
    properties_MC(i).Area = data(3+index_shift);      
    properties_MC(i).Centroid = [data(1+index_shift),data(2+index_shift)];     
    properties_MC(i).MajorAxisLength = data(7+index_shift);  
    properties_MC(i).MinorAxisLength = data(8+index_shift); 
    properties_MC(i).Eccentricity = data(4+index_shift); 
    properties_MC(i).Orientation = 0; 
    properties_MC(i).EquivDiameter = 0; 
    properties_MC(i).Solidity = data(5+index_shift); 
    properties_MC(i).Perimeter = 0; 
    properties_MC(i).WeightedCentroid = 0; 
    properties_MC(i).Circularity = data(6+index_shift); 
    properties_MC(i).EllipticalDeviation = 0; 
    properties_MC(i).MassDisplacement = 0; 
    properties_MC(i).IntegratedIntensity = 0; 
    properties_MC(i).MeanIntensity = data(9+index_shift); 
    properties_MC(i).IntensityDeviation = 0; 
    properties_MC(i).IntensityRange = 0; 
    properties_MC(i).MeanInsideBoundaryIntensity = 0; 
    properties_MC(i).InsideBoundaryIntensityDeviation = 0; 
    properties_MC(i).InsideBoundaryIntensityRange = 0; 
    properties_MC(i).NormalizedInsideBoundaryIntensity = 0; 
    properties_MC(i).MeanOutsideBoundaryIntensity = 0;
    properties_MC(i).OutsideBoundaryIntensityDeviation = 0;
    properties_MC(i).OutsideBoundaryIntensityRange = 0;
    properties_MC(i).NormalizedOutsideBoundaryIntensity = 0;
    properties_MC(i).BoundarySaliency = 0;
    properties_MC(i).NormalizedBoundarySaliency = 0;

end

end

