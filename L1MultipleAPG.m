%function [track_res,output]=L1MultipleAPG(s_frames,paraT)

% This function is used for tracking multiple person using L1 APG method.
% This function created by Atyanta Nika R. 
clc
clear all
close all

%Ambil video
    obj.reader = vision.VideoFileReader('pami.avi');
%Inisialisasi dengan membuat video player untuk output
%obj.videoPlayer = vision.VideoPlayer('Position',[20 400 941 441]);

%Crating Templates for the first time
    frame = obj.reader.step();
    frame = rgb2gray(frame);
    Temp = csvread('InitPos1.csv');
    tsize = [25,15];
%Size all = 24, but minus the 2 referees, it becomes 22 (players only)
    nT = 22;
    nT_ply = 11;
    n_samples       = 600;
    T       = zeros(tsize(1)*tsize(2),nT);
    T_norm  = zeros(1,nT);
    T_mean  = zeros(1,nT);
    T_std   = zeros(1,nT);
    p=cell(1,nT);
    
for i =1:nT,
    p{i}=[Temp(i,1:3);Temp(i,4:6)];
    
    %Templates Player
    [T(:,i),T_norm(i),T_mean(i),T_std(i)] = ...
		corner2image(frame, p{i}, tsize);
end
clear frame
    
    norms   = T_norm.*T_std;            %for showing the templates
    norms_A = T_norm(1:11).*T_std(1:11); %template norms A
    norms_B = T_norm(12:22).*T_std(12:22); %template norms B
    occlusionNf = 0;

%Creating template matrix
dim_T   = size(T,1);
alpha   = 50;

fixT_A   = mean(T(:,1:11),2)/nT_ply;
fixT_B   = mean(T(:,12:22),2)/nT_ply;

A_A       = [T(:,1:11) eye(dim_T)];
Temp_A = [A_A fixT_A]; %becareful
Dict_A = Temp_A'*Temp_A;
Temp1_A = [T(:,1:11) fixT_A]*pinv([T(:,1:11) fixT_A]); %becarefull

A_B       = [T(:,12:22) eye(dim_T)];
Temp_B = [A_B fixT_B]; %becareful
Dict_B = Temp_B'*Temp_B;
Temp1_B = [T(:,12:22) fixT_B]*pinv([T(:,12:22) fixT_B]); %becarefull

T_id	= -(1:nT);

%This is initializing variables -------------------------------------------
rel_std_aff = [0.03,0.0005,0.0005,0.03,1,1];
q = cell(1,nT);
indq = cell(1,nT);
map_aff = cell(1,nT); 
aff_samples = cell(1,nT);
Y = cell(1,nT);
Y_inrange = cell(1,nT);
Y_crop_mean = cell(1,nT);
Y_crop_norm = cell(1,nT);
Y_crop_std = cell(1,nT);
nframes = 0;

%Those Parameters
para.Lambda = [0.2,0.001,10];
para.angle_threshold = 90;
para.Lip	= 8;
para.Maxit	= 10;
para.nT		= 22;%number of templates for the sparse representation
para.rel_std_afnv = [0.03,0.0005,0.0005,0.03,1,1];%diviation of the sampling of particle filter
para.n_samples	= 600;		%number of particles
para.sz_T		= tsize;
para.bDebug		= 1;
%--------------------------------------------------------------------------

tic

%fixT=T;
%Creating Affine samples for cropped image Y
for i=1:22,
    aff_obj = corners2affine(p{i},tsize);
    map_aff{i} = aff_obj.afnv;
    aff_samples{i} = ones(n_samples,1)*map_aff{i};
end


%THE TRACKING
%Reading the data for each frame
while ~isDone(obj.reader)
    img_c = obj.reader.step();
    img = double(rgb2gray(img_c));
    imshow(img_c);hold on
    nframes = nframes+1;
    min_angle = zeros(nT,1);
    %Draw transformation samples from Gaussian Distribution
    for i=1:22
        sc          = sqrt(sum(map_aff{i}(1:4).^2)/2);
        std_aff     = rel_std_aff.*[1 sc sc 1 sc sc];
        map_aff{i}     = map_aff{i} + 1e-14;
        aff_samples{i} = draw_sampleQ(aff_samples{i},std_aff);
        
        %Crop candidate for generating Y
        [Y{i}, Y_inrange{i}] = crop_candidates(im2double(img), aff_samples{i}, tsize);
        
        if (sum(Y_inrange{i}==0) == n_samples)
            sprintf('Target is out of frame!\n');
        end
        
        [Y{i}, Y_crop_mean{i},Y_crop_std{i}] = whitening(Y{i});
        [Y{i}, Y_crop_norm{i}] = normalizeTemplates(Y{i});
        
        qq = zeros(n_samples,1);
        %The calculation of q (the distance of Y and transformed template)
        for j=1:n_samples
            if Y_inrange{i}(j)==0 || sum(abs(Y{i}(:,j)))==0,
                continue;
            end
            
            %L2 norm bounding
            if i<=11,
               qq(j) = norm(Y{i}(:,j)-Temp1_A*Y{i}(:,j));
               qq(j) = exp(-alpha*qq(j)^2);
            else
               qq(j) = norm(Y{i}(:,j)-Temp1_B*Y{i}(:,j));
               qq(j) = exp(-alpha*qq(j)^2);
            end
        end
        [qq,indx] = sort(qq,'descend');
        q{i} = qq;
        indq{i} = indx;
        
        %Ini pembuatan sparse x dengan menggunakan algoritma APGLasso
        n = 1;
        tau = 0;
        Prob = zeros(n_samples,1);
        eta_max = -inf;
        
        while (n<n_samples)&&(q{i}(n)>=tau)
            if i<=11,
                [c] = APGLASSOup(Temp_A'*Y{i}(:,indq{i}(n)),Dict_A,para);
                D_s = (Y{i}(:,indq{i}(n)) - [A_A(:,1:nT_ply) fixT_A]*[c(1:nT_ply); c(end)]).^2;%reconstruction error
            else
                [c] = APGLASSOup(Temp_B'*Y{i}(:,indq{i}(n)),Dict_B,para);
                D_s = (Y{i}(:,indq{i}(n)) - [A_B(:,1:nT_ply) fixT_B]*[c(1:nT_ply); c(end)]).^2;%reconstruction error
            end
            
            Prob(indq{i}(n)) = exp(-alpha*(sum(D_s)));
            tau = tau + Prob(indq{i}(n))/(2*n_samples-1);
            
            if(sum(c(1:nT_ply))<0)
                continue;
            elseif(Prob(indq{i}(n))>eta_max)
                id_max = indq{i}(n);
                c_max  = c;
                eta_max = Prob(indq{i}(n));
                Min_Err(nframes) = sum(D_s);
            end
            n=n+1;
        end
        
        %Resample according to probability
        map_aff{i} = aff_samples{i}(id_max,1:6); %target transformation parameters with the maximum probability
        
        a_max = c_max(1:nT_ply);
    
        [~,indA] = max(a_max);
        [aff_samples{i},~] = resample(aff_samples{i},Prob,map_aff{i});
        if i <=11
            min_angle(i) = images_angle(Y{i}(:,id_max),A_A(:,indA));
            %norms = norms_A;
        else
            min_angle(i) = images_angle(Y{i}(:,id_max),A_B(:,indA));
            %norms = norms_B;
        end
        ratio(nframes) = norm(c_max(nT_ply:end-1));
        Coeff(:,nframes) = c_max;
        
        %track_res{i}(:,nframes)= map_aff{i}';
       
        %This is how I update the templates
        occlusionNf = occlusionNf-1;
        level_bw = 0.03;
        if(min_angle(i) > para.angle_threshold && occlusionNf <0)
            disp(['This is update the template in player: ' num2str(i)]);
            trivial_coef = c_max(nT_ply+1:end-1);
            trivial_coef = reshape(trivial_coef,tsize);
            trivial_coef = im2bw(trivial_coef,level_bw);
            
            se = [0 0 0 0 0;
                0 0 1 0 0;
                0 1 1 1 0;
                0 0 1 0 0;
                0 0 0 0 0;];
            
            trivial_coef = imclose(trivial_coef, se); %This is for filling the hole of image
            cc = bwconncomp(trivial_coef);
            stats = regionprops(cc,'Area');
            areas = [stats.Area];
            
            %Occlusion detection
            if (max(areas) < round(0.25*prod(tsize))),
                [~,indW] = min(a_max(1:nT_ply));
                
                %Instert new template to the minimum value
                T(:,indW) = Y{i}(:,id_max);
                T_mean(:,indW) = Y_crop_mean{i}(id_max);
                T_id(indW) = nframes;
                
                norms(indW) = Y_crop_std{i}(id_max)*Y_crop_norm{i}(id_max);%for showing the templates
                
                if i<=nT_ply,
                    norms_A(indW) = Y_crop_std{i}(id_max)*Y_crop_norm{i}(id_max);
                    [T(:,1:11),~] = normalizeTemplates(T(:,1:11));
                    
                    A_A(:,1:nT_ply) = T(:,1:11); 
                    Temp_A = [A_A fixT_A]; %becareful
                    Dict_A = Temp_A'*Temp_A;
                    Temp1_A = [T(:,1:11) fixT_A]*pinv([T(:,1:11) fixT_A]); %becarefull
                else
                    norms_B(indW) = Y_crop_std{i}(id_max)*Y_crop_norm{i}(id_max);
                    [T(:,12:22),~] = normalizeTemplates(T(:,12:22));
                    
                    A_B(:,1:nT_ply) = T(:,12:22);
                    Temp_B = [A_B fixT_B]; %becareful
                    Dict_B = Temp_B'*Temp_B;
                    Temp1_B = [T(:,12:22) fixT_B]*pinv([T(:,12:22) fixT_B]); %becarefull
                end
            else
                occlusionNf=5;
                para.Lambda(3)=0;
            end
            
        elseif occlusionNf<0
            para.Lambda(3) = 10;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    img_C = double(img_c);
    img_C = showTemplates(img_C,T,T_mean,norms,tsize,nT);
    imshow(img_C);hold on
    text(5,10,num2str(nframes),'FontSize',18,'Color','r');
    thedrawing(map_aff,tsize,min_angle);
    drawnow;
    hold off;
end
Time = toc