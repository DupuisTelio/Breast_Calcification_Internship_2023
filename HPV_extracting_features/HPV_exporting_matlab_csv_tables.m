function HPV_exporting_matlab_csv_tables(inputFilePath,outputFilePath,main_path)
    % This function export matlab csv table to more conventionnal csv table using "stronger" delimiters

    inputFilePath=[main_path '/output' '/features/' inputFilePath '.csv'];
    outputFilePath=[main_path '/output' '/features/' outputFilePath '.csv'];
    
    % Reading csv entry file
    inputData = readtable(inputFilePath);
    
    % Writing data in the output file with the correct delimiter
    writetable(inputData, outputFilePath, 'Delimiter', ';')

    disp(['The table : ' inputFilePath ' has been exported to the new format to: ' outputFilePath]);
end

