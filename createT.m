function [T,T_norm,T_std]=createT(the_path,Location)
    %Location is from saved csv object
    %CSVREAD(LOCATION)
    
    Sz = size(Location,1);
    
    for i=1:Sz
        Img = imread([the_path, num2str(Location(i,1)),'.jpg']);
        Img = rgb2gray(Img);
        p = [Location(i,1) Location(i,2) Location(i,3); Location(i,4) Location(i,5) Location(i,6)];
        [T(:,i),T_norm(i),T_std(i)] = corner2image(Img,p,[Location(i,8),Location(i,9)]);
    end
save('newAccT.mat','T','T_norm','T_std');
end