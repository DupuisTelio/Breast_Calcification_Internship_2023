function main_path = HPV_making_path(main_path)
    % this function is there just to make the path
    
    main_path=['HPV_' main_path];
    % Main folder
    if ~exist(main_path, 'dir')
        mkdir(main_path);
    end
    
    % Subfolder
    input_path= [main_path '/input'];
    output_path= [main_path '/output'];
    if ~exist(input_path, 'dir')
        mkdir(input_path);
    end
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end

    % Sub sub folder inpt
    img_inpt= [input_path '/img'];
    charac_lists_inpt= [input_path '/charac_lists'];
    if ~exist(img_inpt, 'dir')
        mkdir(img_inpt);
    end
    if ~exist(charac_lists_inpt, 'dir')
        mkdir(charac_lists_inpt);
    end

    % Sub sub folder outpt
    img_outpt= [output_path '/img'];
    features_outpt= [output_path '/features'];
    if ~exist(img_outpt, 'dir')
        mkdir(img_outpt);
    end
    if ~exist(features_outpt, 'dir')
        mkdir(features_outpt);
    end
end

