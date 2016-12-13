function [MyFrame]=jajalAty()

clc
clear all
close all


global MyFrame
global HitungG
HitungG = 1;

%ini untuk menginisialisasi inputan, untuk membuat sistem obyek, mendeteksi
%obyek foreground dan mendisplay hasilnya.

obj.reader = vision.VideoFileReader('pami2.avi');

%membuat dua video player yang satu untuk mask dan yang lainnya untuk
%memainkan foreground mask.

obj.videoPlayer = vision.VideoPlayer('Position',[20,400,700,400]);
obj.maskPlayer = vision.VideoPlayer('Position',[740,400,700,400]);

obj.detector = vision.ForegroundDetector('NumGaussians',3,'NumTrainingFrames',300,'MinimumBackgroundRatio',0.7,'InitialVariance',0.19^2); %std = 0.1627
obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort',true,'AreaOutputPort',true,'CentroidOutputPort',true,'MinimumBlobArea',7);

tracks = struct('id',{},'bbox',{},'kalmanFilter',{},'age',{},'totalVisibleCount',{},'consecutiveInvisibleCount',{});

nextId = 1;
NoOfFrame = 1;
FrameCountainer = [];
while ~isDone(obj.reader)
    frame=obj.reader.step();
    frame=imadjust(frame,[0.1 0.09 0.1;1 0.9 1],[0.2 0.2 0.2;1 1 1]);
    FrameContainer(:,:)=rgb2gray(frame);
    
    MyFrame(NoOfFrame).TheMean = mean2(FrameContainer);
    MyFrame(NoOfFrame).TheStd = std2(FrameContainer);
    
    %Detect Foreground
    mask = obj.detector.step(frame); %using gaussian mixture model
    mask = medfilt2(mask);
    %figure(2),imshow(mask),title('THE MASK');
    %Apply morphological operation to remove noise and fill in holes.
    mask = imopen(mask, strel('rectangle',[3,3]));
    mask = imclose(mask, strel('rectangle',[6,6]));
    mask = imfill(mask, 'holes');
    
    %imshow(mask)
    
    [~,centroids,bboxes] = obj.blobAnalyser.step(mask);
    [centroids,bboxes]=colorChecking(centroids,bboxes,frame,mask);
    %[~,centroids,bboxes] = obj.blobAnalyser.step(mask);
    
    [noTrack,~]=size(bboxes);
    disp(['Jumlah setan(penampakan) yang terlihat= ',num2str(noTrack),' buah']);
    predictNewLocationsOfTracks();
    
    nTracks = length(tracks);
    nDetections = size(centroids,1);
    
    %Compute the cost of assigning each detection to each track
    cost = zeros(nTracks,nDetections);
    for i=1:nTracks
        cost(i,:) = distance(tracks(i).kalmanFilter,centroids);
    end
    
    %Solve the assignment problem
    costOfNonAssignment = 10;
    [assignments,unassignedTracks,unassignedDetections] = assignDetectionsToTracks(cost,costOfNonAssignment); %This is james munkres Hungarian algorithm
    numAssignedTracks = size(assignments,1);
    for i=1:numAssignedTracks,
        trackIdx = assignments(i,1);
        detectionIdx = assignments(i,2);
        centroid = centroids(detectionIdx,:);
        bbox = bboxes(detectionIdx,:);
        correct(tracks(trackIdx).kalmanFilter,centroid);
        tracks(trackIdx).bbox = bbox;
        tracks(trackIdx).age = tracks(trackIdx).age+1;
        tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount +1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
    end
    for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount + 1;
    end
   if ~isempty(tracks)
        invisibleForTooLong = 10;
        ageThreshold = 10;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.4) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
   end
   centroids = centroids(unassignedDetections, :);
   bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
                centroid = centroids(i,:);
                bbox = bboxes(i, :);
            
                % Create a Kalman filter object.
                kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                               centroid, [200, 50], [100, 25], 100);
            
                % Create a new track.
                newTrack = struct(...
                          'id', nextId, ...
                          'bbox', bbox, ...
                          'kalmanFilter', kalmanFilter, ...
                          'age', 1, ...
                          'totalVisibleCount', 1, ...
                          'consecutiveInvisibleCount', 0);
            
                % Add it to the array of tracks.
                tracks(end + 1) = newTrack;
            
                % Increment the next id.
                nextId = nextId + 1;
        end     

NoOfFrame = NoOfFrame+1;
displayTrackingResult()  
end


function predictNewLocationsOfTracks()
    for i=1:length(tracks)
        bbox = tracks(i).bbox;
        predictedCentroid = predict(tracks(i).kalmanFilter);
        predictedCentroid = int32(predictedCentroid)-bbox(3:4)/2;
        tracks(i).bbox = [predictedCentroid, bbox(3:4)];
    end
end

    function displayTrackingResult()
        frame = im2uint8(frame);
        mask = uint8(repmat(mask,[1,1,3])).*255;
        
        minVisibleCount = 8;
        if ~isempty(tracks)
            reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            if ~isempty(reliableTracks)
                bboxes = cat(1,reliableTracks.bbox);
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' prediksi'};
                labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end
        obj.maskPlayer.step(mask);        
        obj.videoPlayer.step(frame);
    end
end

function [centroidsN,bboxesN]=colorChecking(centroids,bboxes,frame,mask)
    global HitungG
    Hitung = 1;
    if ~isempty(bboxes)||~isempty(centroids),
    
        bboxesN = [];
        centroidsN = [];
        ObjectLocationXY = zeros(4,2,size(bboxes,1));
        for i=1:size(bboxes,1),
            ObjectLocationXY(1,:,i) = bboxes(i,1:2);
            ObjectLocationXY(2,:,i) = [bboxes(i,1),bboxes(i,2)+bboxes(i,4)];
            ObjectLocationXY(3,:,i) = [bboxes(i,1)+bboxes(i,3),bboxes(i,2)];
            ObjectLocationXY(4,:,i) = [bboxes(i,1)+bboxes(i,3),bboxes(i,2)+bboxes(i,4)];
            ObjectArea = zeros(size(frame(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i))));
            ObjectArea = frame(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i),:);
            ObjectAreaBin(:,:,1) = mask(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i));
            ObjectAreaBin(:,:,2) = mask(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i));
            ObjectAreaBin(:,:,3) = mask(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i));
            ObjectArea = ObjectArea .* ObjectAreaBin;
            ObjectAreaBin=[];
            %imwrite(ObjectArea,['D:\03 School\Buku_Buku\15 Computer Vision\MultiObjectTracking\Bayangan Hitam\',num2str(HitungG),'.jpg'],'jpg');
            try
                ObjectArea = rgb2hsv(ObjectArea);
                meanHSV = mean2(ObjectArea(:,:,3));
                Y = 1/size(bboxes,1)*ItungProb(meanHSV,0.1231,0.0123);
            catch
                Y = 0.6
            end
            
            %HitungG = HitungG+1;
            if Y < 0.05
                %bboxes(i,:) = zeros(1,4);
                %centroids(i,:) = zeros(size(centroids(i,:)));
                bboxesN(Hitung,:)=bboxes(i,:);
                centroidsN(Hitung,:)=centroids(i,:);
                Hitung = Hitung+1;
            else
                figure(3),imshow(frame(ObjectLocationXY(1,2,i):ObjectLocationXY(2,2,i),ObjectLocationXY(1,1,i):ObjectLocationXY(3,1,i),:)),title('Bayangan');
            end
        end
        bboxesN=int32(bboxesN);
        centroidsN=int32(centroidsN);
    else
        centroidsN = centroids;
        bboxesN = bboxes;
    end
end