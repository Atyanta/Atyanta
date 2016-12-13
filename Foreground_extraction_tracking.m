%Ini adalah perbandingan tracking Matlab menggunakan Foreground detector

hfg = vision.ForegroundDetector(...
        'NumTrainingFrames', 455, ... % 5 because of short video
        'InitialVariance', 30*30); % initial standard deviation of 30
    
hblob = vision.BlobAnalysis(...
        'CentroidOutputPort', false, 'AreaOutputPort', false, ...
        'BoundingBoxOutputPort', true, 'MinimumBlobArea', 20);

hsnk = vision.VideoPlayer();

n=1;

    while n <456
      PATH   = ['D:\03 School\Buku_Buku\15 Computer Vision\Sparse Learning\L1-APG_release\Bola2\', sprintf(['%04d'],n),'.jpg'];
      frame  = imread(PATH);
      fgMask = step(hfg, frame);
      bbox   = step(hblob, fgMask);
      
      % draw bounding boxes around cars
      out    = insertShape(frame, 'Rectangle', bbox, 'Color', 'White');
      step(hsnk, out); % view results in the video player
      n=n+1;
    end