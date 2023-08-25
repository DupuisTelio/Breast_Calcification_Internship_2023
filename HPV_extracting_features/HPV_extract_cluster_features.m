function HPV_extract_cluster_features(table_name,file_name,database,set_feature,set_feature_name,main_path) 
    % This function can write in an existing csv table (or create a new one if not existing table) the clusters features (such as overlap, intersected area,...) with as 
    % a first identifer the file_name of the  mammogram (more frequently available than the patient_id) and the dataset that the mammogram belongs to.
    % Each line of the table is a different mammogram
    
    path = [main_path '/output' '/features/' table_name '.csv'];
    if exist(path, 'file')
        % Open csv table on append mode
        tableID = fopen(path, 'a');
    else
        tableID = fopen(path, 'w');
        % Write the header (first line) in the CSV file
        fprintf(tableID, 'Database-and-File_name,%s\n', strjoin(set_feature_name, ','));
    end
    
    long_file_name= [database '_' file_name];

    % Convert set_feature to a cell array of strings
    % Convert numeric values to strings
    set_feature_str = arrayfun(@num2str, set_feature, 'UniformOutput', false);

    % Write the values for the ID specified in the CSV file
    fprintf(tableID, '%s,%s\n', long_file_name, strjoin(set_feature_str, ','));
    
    % Close the table
    fclose(tableID);
    
    disp(['Clusters Features data has been added to : ' path]);
end
