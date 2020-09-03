% Following: Read facies model from file
faciesfileID = fopen('E:\CNN_ModelingProject\Data and Results\Unconditional GAN\2-Progressive GAN\Partial assessement of global features\images_generated_300.out', 'r');
facies_no = 300;
allfacies = fscanf(faciesfileID, '%f \n', [64, 64*facies_no]);       %imshow(facies);
allfacies = permute(reshape(allfacies,[64, 64, facies_no]),[2 1 3]);

startpoints = zeros(facies_no, 2);
endpoints = zeros(facies_no, 2);
reallengths = zeros(facies_no,1);
widths = zeros(facies_no,1);

for i =1: facies_no
    i
    facies = allfacies(:,:, i);
    facies =  imbinarize(facies);
    % Following: Mannually select one connected channel
    [facies2] = bwselect(facies,4);
    % Following: Calculate real length and Euclidean distance of selected channel
    channel_skl = single(bwskel(facies2));  % skeleton of channel
    [endx, endy] = find(bwmorph(channel_skl,'endpoints')); % endpoints of skeleton
    startpoint = [endx(1,1), endy(1,1)];
    endpoint = [endx(2,1), endy(2,1)];
    QDM = bwdistgeodesic(logical(channel_skl), [endy(1,1)], [endx(1,1)], 'quasi-euclidean');  % Quasi-euclidean distance matrix of channel pixels from the start point of channel
    reallength = QDM(endx(2,1), endy(2,1));
    % Following: Calculate width of channel
    disttobdy = bwdist(1-facies2);
    width = sum(disttobdy.*channel_skl,'all')/sum(channel_skl, 'all');
    %Following store calculated parameters
    startpoints(i,:) = startpoint;
    endpoints(i,:) = endpoint;
    reallengths(i,1) = reallength;
    widths(i,1) = width;
end

