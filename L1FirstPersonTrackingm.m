clc
%clear all
close all


%% parameters setting for tracking
para.lambda = [0.2,0.001,10]; % lambda 1, lambda 2 for a_T and a_I respectively, lambda 3 for the L2 norm parameter
% set para.lambda = [a,a,0]; then this the old model
para.Lambda = [0.2,0.001,10];
para.angle_threshold = 70;
para.Lip	= 8;
para.Maxit	= 11;
para.nT		= 377;%number of templates for the sparse representation
para.rel_std_afnv = [0.03,0.0005,0.0005,0.03,1,1];%diviation of the sampling of particle filter
para.n_sample	= 600;		%number of particles
para.sz_T		= [30,15];
%para.init_pos	= init_pos;
para.bDebug		= 1;	   %debugging indicator
bShowSaveImage	= 1;       %indicator for result image show and save after tracking finished
%para.s_debug_path = res_path;
%----------------------------------------------------------------------------------------------


obj.reader = vision.VideoFileReader('pami2.avi');
img = obj.reader.step();
img = rgb2gray(img);

Temp = csvread('InitPos.csv');
T = zeros(prod(para.sz_T),22);
%[T,T_norm,T_mean,T_std] = InitTemplates(para.sz_T,para.nT,img,para.init_pos);
% for i=1:22  
%     IP{i} = [Temp(i,1) Temp(i,2) Temp(i,3);Temp(i,4) Temp(i,5) Temp(i,6)];
%         [T(:,i),T_norm(i),T_mean(i),T_std(i)] = ...
% 		corner2image(img, IP{i}, para.sz_T);
% end

the_path = 'D:\03 School\Buku_Buku\15 Computer Vision\MultiObjectTracking\';
Location = csvread([the_path,'Location.csv']);

[T,T_norm,T_std] = createT(the_path,Location);


para.init_pos	= [Temp(1,1) Temp(1,2) Temp(1,3);Temp(1,4) Temp(1,5) Temp(1,6)];

norms = T_norm.*T_std; %template norms
occlusionNf = 0;

%L1 Function settings
dim_T	= size(T,1);	%number of elements in one template, sz_T(1)*sz_T(2)=12x15 = 180
A		= [T eye(dim_T)]; %data matrix is composed of T, positive trivial T.
alpha = 50;%this parameter is used in the calculation of the likelihood of particle filter
aff_obj = corners2affine(para.init_pos, para.sz_T); %get affine transformation parameters from the corner points in the first frame
map_aff = aff_obj.afnv;
aff_samples = ones(para.n_sample,1)*map_aff;

T_id	= -(1:para.nT);	% template IDs, for debugging
fixT = T(:,1)/para.nT; % first template is used as a fixed template

%Temaplate Matrix
Temp = [A fixT];
Dict = Temp'*Temp;
Temp1 = [T,fixT]*pinv([T,fixT]);


%%Tracking
rel_std_afnv = para.rel_std_afnv;
% initialization
nframes	= 0;


while ~isDone(obj.reader)
    nframes = nframes+1;
    img_c = obj.reader.step();
    img = double(rgb2gray(img_c));
    %-Draw transformation samples from a Gaussian distribution
    sc			= sqrt(sum(map_aff(1:4).^2)/2);
    std_aff		= rel_std_afnv.*[1, sc, sc, 1, sc, sc];
    map_aff		= map_aff + 1e-14;
    aff_samples = draw_sample(aff_samples, std_aff); %draw transformation samples from a Gaussian distribution
    %-Crop candidate targets "Y" according to the transformation samples
    [Y, Y_inrange] = crop_candidates(im2double(img), aff_samples(:,1:6), para.sz_T);
    if(sum(Y_inrange==0) == para.n_sample)
        sprintf('Target is out of the frame!\n');
    end
    
    [Y,Y_crop_mean,Y_crop_std] = whitening(Y);	 % zero-mean-unit-variance
    [Y, Y_crop_norm] = normalizeTemplates(Y); %norm one
    
    %-L1-LS for each candidate target
    eta_max	= -inf;
    q   = zeros(para.n_sample,1); % minimal error bound initialization
   
    % first stage L2-norm bounding    
    for j = 1:para.n_sample
        if Y_inrange(j)==0 || sum(abs(Y(:,j)))==0
            continue;
        end
        
        % L2 norm bounding
        q(j) = norm(Y(:,j)-Temp1*Y(:,j));
        q(j) = exp(-alpha*q(j)^2);
    end
    %  sort samples according to descend order of q
    [q,indq] = sort(q,'descend');    
    
    % second stage
    p	= zeros(para.n_sample,1); % observation likelihood initialization
    n = 1;
    tau = 0;
    while (n<para.n_sample)&&(q(n)>=tau)        

        [c] = APGLASSOup(Temp'*Y(:,indq(n)),Dict,para);
        D_s = (Y(:,indq(n)) - [A(:,1:para.nT) fixT]*[c(1:para.nT); c(end)]).^2;%reconstruction error                
        p(indq(n)) = exp(-alpha*(sum(D_s))); % probability w.r.t samples
        tau = tau + p(indq(n))/(2*para.n_sample-1);%update the threshold
        
        if(sum(c(1:para.nT))<0) %remove the inverse intensity patterns
            continue;
        elseif(p(indq(n))>eta_max)
            id_max	= indq(n);
            c_max	= c;
            eta_max = p(indq(n));
            Min_Err(nframes) = sum(D_s);
        end
        n = n+1;
    end
   % figure,stem(p);
    count(nframes) = n;    
    
    % resample according to probability
    map_aff = aff_samples(id_max,1:6); %target transformation parameters with the maximum probability
    a_max	= c_max(1:para.nT);
    [aff_samples, ~] = resample(aff_samples,p,map_aff); %resample the samples wrt. the probability
    [~, indA] = max(a_max);
    min_angle = images_angle(Y(:,id_max),A(:,indA));
    ratio(nframes) = norm(c_max(para.nT:end-1));
    Coeff (:,nframes) = c_max;
    
    
     %-Template update
     occlusionNf = occlusionNf-1;
     level = 0.03;

        if( min_angle > para.angle_threshold && occlusionNf<0 )        
            disp('Update!')
            trivial_coef = c_max(para.nT+1:end-1);
            trivial_coef = reshape(trivial_coef, para.sz_T);
        
            trivial_coef = im2bw(trivial_coef, level);

            se = [0 0 0 0 0;
                0 0 1 0 0;
                0 1 1 1 0;
                0 0 1 0 0'
                0 0 0 0 0];
            trivial_coef = imclose(trivial_coef, se);
        
            cc = bwconncomp(trivial_coef);
            stats = regionprops(cc, 'Area');
            areas = [stats.Area];
        
            % occlusion detection 
            if (max(areas) < round(0.25*prod(para.sz_T)))        
                % find the tempalte to be replaced
                [~,indW] = min(a_max(1:para.nT));
        
                % insert new template
                T(:,indW)	= Y(:,id_max);
                T_mean(indW)= Y_crop_mean(id_max);
                T_id(indW)	= nframes; %track the replaced template for debugging
                norms(indW) = Y_crop_std(id_max)*Y_crop_norm(id_max);
        
                [T, ~] = normalizeTemplates(T);
                A(:,1:para.nT)	= T;
        
                %Temaplate Matrix
                Temp = [A fixT];
                Dict = Temp'*Temp;
                Temp1 = [T,fixT]*pinv([T,fixT]);
            else
                occlusionNf = 5;
                % update L2 regularized term
                para.Lambda(3) = 0;
            end
        elseif occlusionNf<0
            para.Lambda(3) = 10;
        end
     
    %Time_record(nframes) = toc;
    
    
    % The dormitory
        img_color	= double(img_c);
%        img_color	= showTemplates(img_color, T, T_mean, norms, para.sz_T, para.nT);
        imshow(img_color);
        text(5,10,num2str(nframes),'FontSize',14,'Color','r');
        text(5,30,['The angle: ',num2str(min_angle)],'FontSize',12,'Color','b');
        text(5,50,['Tracked Prob: ',num2str(eta_max)],'FontSize',12,'Color','b');
        
             if eta_max < 1.5e-20
                text(5,90,['LOST TRACK....'],'FontSize',20,'Color','r');
             end
        
        color = [1 0 0];
        drawAffine(map_aff, para.sz_T, color, 2);
        drawnow;
    
end

%The parameterss
