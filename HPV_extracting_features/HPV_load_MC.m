function MC_locations = HPV_load_MC(filename)



%% Load the data from the text file
data = load(filename); 

k = 2;
j = 1;
CellTab = cell(1, 1); % Init cell tab un tableau de cellules

while k <= length(data)
    if data(k) == 0 % start of a new MC
        k = k + 1;
        j = j + 1;
    else
        i = 1;
        while data(k) ~= 0 % cross whole MC
            CellTab{j}(i, :) = [data(k), data(k+1)]; % add a new line, double format
            i = i + 1;
            k = k + 2;
        end
    end
end

%Tranpose
MC_locations=transpose(CellTab);


end


