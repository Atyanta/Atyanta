function ProveThatIamRight(aff_samples,tsize)
%This program is proving that I'm right, where particle filter is used to
%sample the image, then the result is being used for tracking.


%load('AffSamples.mat');
%paths = 'D:\03 School\Buku_Buku\15 Computer Vision\Sparse Learning\L1-APG_release\car4\001.jpg';

%its name is aff_samples (size 600x6)

%img = imread('D:\03 School\Buku_Buku\15 Computer Vision\Sparse Learning\L1-APG_release\car4\0001.jpg');
%obj.reader = vision.VideoFileReader('pami.avi');
%img = obj.reader.step();
%imshow(img);hold on

[sizeRow,~] = size(aff_samples);

R = zeros(3,3);
dummy = zeros(1,6);
Outs = [ 1 tsize(1) 1;1 1 tsize(2);1 1 1];
InitPos = zeros(2,3);

for i=1:sizeRow,
    dummy=aff_samples(i,:);
    R = [dummy(1,1) dummy(1,2) dummy(1,5);...
         dummy(1,3) dummy(1,4) dummy(1,6);...
         0          0          1];
    Position = round(R*Outs);
    InitPos = Position(1:2,:);
    nX = mean([InitPos(2,1) InitPos(2,3)]);
    nY = mean([InitPos(1,1) InitPos(1,2)]);
    plot(nX,nY,'*r');hold on
    %plot([InitPos(2,:) InitPos(2,2)+(abs(InitPos(2,1)-InitPos(2,3)))],[InitPos(1,:) InitPos(1,2)],'g');hold on
    %YY = [InitPos(2,2) InitPos(2,1) InitPos(2,3) InitPos(2,2)+abs(InitPos(2,3)-InitPos(2,1)) InitPos(2,2)];
    %XX = [InitPos(1,2) InitPos(1,1) InitPos(1,3) InitPos(1,2) InitPos(1,2)];
    %plot(YY,XX,'g'); hold on
end
%hold off