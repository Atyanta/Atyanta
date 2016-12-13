clc
clear all
close all

Obj_path = 'D:\03 School\Buku_Buku\15 Computer Vision\Sparse Learning\L1-APG_release\car4\0001.jpg';

img = imread(Obj_path);
imshow(img)
hold on
init_pos= [55,140,53;65,64,170];
y = [140 55 53];
x = [64 65 170];
plot(x,y,'r--o','LineWidth',1)
hold on

% AKU BIKIN SEMBARANG RANDOM VARIABLE DARI 0-1
warna = ['y--*' 'm--*' 'c--*' 'r--*' 'g--*' 'b--*' 'k--*' 'y--d' 'm--d'];

p = cell(1,10);
p{1}=init_pos;
for i=2:10,
    p{i} = init_pos + rand(2,3)*0.6;
    y = [p{i}(1,2) p{i}(1,1) p{i}(1,3)];
    x = [p{i}(2,2) p{i}(2,1) p{i}(2,3)];
    plot(x,y,warna(i-1))
    hold on
end

T = zeros(180,10);
for i=1:10,
    R = [p{i}(:,:); ones(1,3)]/([1 12 1;1 1 15;1 1 1]);
    map_afnv = [R(1,1) R(1,2) R(2,1) R(2,2) R(1,3) R(2,3)];
    img_map = IMGaffine_c(double(img),map_afnv,[12,15]);
    in = reshape(img_map,180,1);
    outs = (in-ones(180,1)*mean(in))./(ones(180,1)*(std(in)+1e-14));
    T(:,i) = outs/norm(outs);
end

%Calculating each Frame for getting Y
%Mempersiapkan Aff_samples dari Draw samples
%1. Mempersiapkan Aff_samples
R = [p{1}(:,:);ones(1,3)]/([1 12 1;1 1 15;1 1 1]);
map_afnv = [R(1,1) R(1,2) R(2,1) R(2,2) R(1,3) R(2,3)];
aff_samples = ones(600,1)*map_afnv;

%2. Mempersiapkan Std_afnv.
sc = sqrt(sum(map_afnv(1:4).^2)/2);
std_afnv = [0.03,0.0005,0.0005,0.03,1,1].*[1 sc sc 1 sc sc];

%3. Menghitung draw sample
Out = zeros(600,6);
aff_samples(:,1) = log(aff_samples(:,1));
aff_samples(:,4) = log(aff_samples(:,4));
Out(:,1:6) = randn([600,6])*diag(std_afnv) + aff_samples;
Out(:,1) = exp(Out(:,1));
Out(:,4) = exp(Out(:,4));

[Y,Y_inrange] = crop_candidates(im2double(img),aff_samples(:,1:6), [12 15]);
[Y,Y_crop_mean,Y_crop_std] = whitening(Y);	 % zero-mean-unit-variance
[Y, Y_crop_norm] = normalizeTemplates(Y); %norm one