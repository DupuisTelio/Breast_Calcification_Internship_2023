function HPV_extract_cluster_index(output_table_name,data2cluster,nuclei,main_path)
    % This function extract the index of the clusters each points belong to (can be used later on for the python code) in a csv table

    % Open file on writing mode
    path=[main_path '/output' '/features/' output_table_name '.csv'];
    file = fopen(path, 'w');
    
    % Write CSV file header
    fprintf(file, 'MC_index,Cluster,X,Y\n');
    
    % For each MC
    for mc_index = 1:numel(nuclei)

        mc_data = nuclei{mc_index};
        num_pts = size(mc_data, 1);
        cluster_index=data2cluster(mc_index); % data2cluster - for every data point which cluster it belongs to (numPts)

        % For each points of the MC
        for pt_index = 1:num_pts
            x = mc_data(pt_index, 2);
            y = mc_data(pt_index, 1);
            
            % Write the data in the csv file
            fprintf(file, '%d,%d,%.6f,%.6f\n', mc_index, cluster_index, x, y);
        end
    end
    
    % Fermer le fichier
    fclose(file);

    disp(['Clusters Index data table has been created to : ' path]);
end

