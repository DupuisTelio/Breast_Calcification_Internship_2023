function HPV_extract_cluster_centers(output_table_name,clustCent,main_path)
    % this function extract the centroids of each cluster in a csv table

    % clustCent - nuclei cluster centroid that based on nuclei centroid

    % path
    path=[main_path '/output' '/features/' output_table_name '.csv'];

    % Cluster index
    NumCluster = size(clustCent, 2);
    clusterIndices = (1:NumCluster)';
    
    % Concatenate
    dataMatrix = [clusterIndices, clustCent'];

    % header
    header = {'ClusterIndex', 'Value 1: x?', 'Value2', 'Value3:y?', 'Value4'}; 
    
    % Create a table from the data matrix
    dataTable = array2table(dataMatrix, 'VariableNames', header);

    % Write in the csv file
    writetable(dataTable,path);

    disp(['Clusters centroids data table has been created to : ' path]);
end






