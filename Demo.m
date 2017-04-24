%% LOADING DATA
clear;
clc;

%FASTER RCNN WAS USED FOR THE DETECTION OF THE PEDESTRIANS
bboxes.bboxes=load('town_faster_rcnn');

%FEATURELAYER = 'FC8' FOR "IMAGENET-CAFFE-ALEX" WAS USED TO EXTRACT THE
%FEATURES OF EACH DETECTIONS
load ('town_feature_stack.mat')
load ('town_test_stack.mat')

%INITIALIZATION OF THE PARAMETERS
check_previous_stack{251}=1;
check_next_stack{251}=1;
acti_next_stack_res{251}=1;
acti_stack_res{251}=1;
A = single(zeros(1,1000));
B = single(zeros(1,2));
average_activation=zeros(1000,250);
next_id = length(bboxes.bboxes.ans{1, 1});
next_id1 = length(bboxes.bboxes.ans{1, 1});
for ooo = 1:250
all_index{ooo} = 0;
end

%This is the folder containing the frames of the dataset, consider it
%changing to the folder where frames are located. In case of a video file
%it should be converted to individual frames first before runnin the code
imageNames = dir(fullfile('C:\Users\acharyad\Documents\MATLAB\DeepLearningForComputerVision\data','town','*.jpg'));
imageNames = {imageNames.name}';
dataFolder = 'C:\Users\acharyad\Documents\MATLAB\DeepLearningForComputerVision\data';


weight = 0.5; % weight of the pixel and feature vector
thr_fv= 0.8; % threshold of the distance for feature vector
thr_pxl=0.12; % Threshold for the pixel distance
frames_to_compare = 10; % number of frames to compare for reidentification

figure;

%%
for ii = 1:length(imageNames)
%CENTROIDS OF THE PREDICTIONS
cen=[(bboxes.bboxes.ans{1,ii}(:,1)+bboxes.bboxes.ans{1,ii}(:,3))/2, (bboxes.bboxes.ans{1,ii}(:,2)+bboxes.bboxes.ans{1,ii}(:,4))/2];
cen_nf=[(bboxes.bboxes.ans{1,ii+1}(:,1)+bboxes.bboxes.ans{1,ii+1}(:,3))/2, (bboxes.bboxes.ans{1,ii+1}(:,2)+bboxes.bboxes.ans{1,ii+1}(:,4))/2];
    
 %%   READING THE FIRST FRAME AND NEXT FRAME
im = imread(fullfile('C:\Users\acharyad\Documents\MATLAB\DeepLearningForComputerVision\data','town',imageNames{ii}));
im1 = imread(fullfile('C:\Users\acharyad\Documents\MATLAB\DeepLearningForComputerVision\data','town',imageNames{ii+1}));
   
%REARRANGING THE DATA
box = [bboxes.bboxes.ans{1,ii}(:,1), bboxes.bboxes.ans{1,ii}(:,2), (bboxes.bboxes.ans{1,ii}(:,3)-bboxes.bboxes.ans{1,ii}(:,1)), (bboxes.bboxes.ans{1,ii}(:,4)-bboxes.bboxes.ans{1,ii}(:,2))];
box_nf = [bboxes.bboxes.ans{1,ii+1}(:,1), bboxes.bboxes.ans{1,ii+1}(:,2), (bboxes.bboxes.ans{1,ii+1}(:,3)-bboxes.bboxes.ans{1,ii+1}(:,1)), (bboxes.bboxes.ans{1,ii+1}(:,4)-bboxes.bboxes.ans{1,ii+1}(:,2))];

%% NORMALIZED FEATURE VECTOR DISTANCES

D1 = zeros(length(box), length(box_nf));
D2 = zeros(length(box),1);
D3 = zeros(length(box_nf),1);
D4 = zeros(length(box), length(box_nf));

    for imfeat = 1:length(box)
        for timfeat=1:length(box_nf)
            D1(imfeat,timfeat) = pdist2(acti_stack_image{ii}(:,imfeat)',acti_stack{ii}(:,timfeat)');
        end
    end
    for imfeat1 = 1:length(box)
        D2(imfeat1) = pdist2(A, acti_stack_image{ii}(:,imfeat1)');
    end
    for timfeat1 = 1:length(box_nf)
        D3(timfeat1) = pdist2(A, acti_stack{ii}(:,timfeat1)');
    end
    for imfeat2 = 1:length(box)
        for timfeat2=1:length(box_nf)
            D4(imfeat2,timfeat2) = (D1(imfeat2,timfeat2))/ (sqrt(D2(imfeat2)*D3(timfeat2)));
        end
    end

%% NORMALIZED PIXEL DISTANCES

d1 = zeros(length(box), length(box_nf));
d2 = zeros(length(box),1);
d3 = zeros(length(box_nf),1);
d4 = zeros(length(box), length(box_nf));
d = zeros(length(box_nf),1);
n = zeros(length(box_nf),1);

    for imfeat = 1:length(box)
        for timfeat=1:length(box_nf)
            d1(imfeat,timfeat) = pdist2(cen(imfeat,:),cen_nf(timfeat,:));
        end
    end
    for imfeat1 = 1:length(box)
        d2(imfeat1) = pdist2(B, cen(imfeat1,:));
    end
    for timfeat1 = 1:length(box_nf)
        d3(timfeat1) = pdist2(B, cen_nf(timfeat1,:));
    end
    for imfeat2 = 1:length(box)
        for timfeat2=1:length(box_nf)
            d4(imfeat2,timfeat2) = (d1(imfeat2,timfeat2))/ (sqrt(d2(imfeat2)*d3(timfeat2)));
        end
    end
  
    for norm = 1:length(box_nf)
        [d(norm), n(norm)] = min(d4(:,norm));
    end
    
%% COMBINING FEATURE VECTOR DISTANCES WITH PIXEL DISTANCES
Dist_min_norm = zeros(length(box_nf),1);
Index_min_norm = zeros(length(box_nf),1);

comb = zeros(length(box), length(box_nf));
    for imfeat = 1:length(box)
        for timfeat=1:length(box_nf)
            comb(imfeat,timfeat) = ((1-weight)*D4(imfeat,timfeat) + weight*(d4(imfeat,timfeat)));
        end
    end
    for norm = 1:length(box_nf)
       [Dist_min_norm(norm), Index_min_norm(norm)] = min(comb(:,norm));
    end
    
%% RE-IDENTIFICATION IN SUBSEQUENT FRAMES
%Mapping IDs of previous frame to new frame
if ii>1
    Index_min_norm_new = zeros(length(Index_min_norm),1);
    for ppp = 1:length(Index_min_norm)
           Index_min_norm_new(ppp) = Index_min_norm_previous(Index_min_norm(ppp)); 
    end
end


if ii==1
    Index_min_norm_new=Index_min_norm;
end
Index_stack{ii}=Index_min_norm_new;
dist_stack{ii}=Dist_min_norm;


%% CHECKING (FRAMES TO COMPARE) PREVIOUS FRAMES FOR A MATCH
if ii>1
lol=1;
for qqq = 1:length(Index_min_norm_previous)
    if ~(ismember(qqq, Index_min_norm));
        res_acti(lol) = qqq;
        lol=lol+1;
    end
end
end

if exist ('res_acti', 'var');
   check_previous = zeros(1000, length(res_acti));
   acti_stack_res{ii}=res_acti;
for poll = 1:length(res_acti)
   check_previous(:,poll) = acti_stack{ii-1}(:,res_acti(poll));
end
check_previous_stack{ii}=check_previous;
end

toll=1;
for lok = 1:length(Index_min_norm)
    for loki = lok+1:length(Index_min_norm)
        if Index_min_norm(lok)==Index_min_norm(loki)
            if comb(Index_min_norm(lok),lok)<comb(Index_min_norm(lok),loki)
                res_acti_next(toll) = loki;
                toll=toll+1;
            else
                res_acti_next(toll) = lok;
                toll=toll+1;
            end
        end      
    end
end

if exist ('res_acti_next', 'var');
    res_acti_next = unique(res_acti_next);
    acti_next_stack_res{ii} = res_acti_next;
    check_next = zeros(1000, length(res_acti_next));
    for poll = 1:length(res_acti_next)
        check_next(:,poll) = acti_stack{ii}(:,res_acti_next(poll));
    end

check_next_stack{ii}=check_next;
end

if ii>1
    if acti_next_stack_res{ii}
        dist_second = zeros(length(acti_next_stack_res{ii}),1);
        dist_third = zeros(length(acti_next_stack_res{ii}),1);
        Index_second = zeros(length(acti_next_stack_res{ii}),1);
        Index_third = zeros(length(acti_next_stack_res{ii}),1);
        for norm = 1: length(acti_next_stack_res{ii})
                DDD5=sort(comb(:,acti_next_stack_res{ii}(norm)));
                dist_second(norm)=(DDD5(2));
                dist_third(norm)=(DDD5(3));
                Index_second(norm)= find (comb(:,acti_next_stack_res{ii}(norm))==dist_second(norm));
                Index_third(norm)= find (comb(:,acti_next_stack_res{ii}(norm))==dist_third(norm));
                
                if dist_third(norm)< 0.5 && d4(Index_third(norm), acti_next_stack_res{ii}(norm)) < thr_pxl
                    Index_min_norm_new(acti_next_stack_res{ii}(norm))= Index_min_norm_previous(Index_third(norm));
                end
                
                if dist_second(norm)< 0.5 && d4(Index_second(norm), acti_next_stack_res{ii}(norm)) < thr_pxl
                    Index_min_norm_new(acti_next_stack_res{ii}(norm))= Index_min_norm_previous(Index_second(norm));
                end

        end 
    end
    
end

min_dist = thr_fv;
for gulla = 1:frames_to_compare
 
if ii>gulla+1 && ~isempty(check_previous_stack{1,ii-gulla}) && ~isempty(check_next_stack{1,ii})
    
Df1 = zeros(length(check_previous_stack{1,ii-gulla}(1,:)), length(check_next_stack{1,ii}(1,:)));
Df2 = zeros(length(check_previous_stack{1,ii-gulla}(1,:)),1);
Df3 = zeros(length(check_next_stack{1,ii}(1,:)),1);
Df4 = zeros(length(check_previous_stack{1,ii-gulla}(1,:)), length(check_next_stack{1,ii}(1,:)));
Dist_frame = zeros(length(check_next_stack{1,ii}(1,:)),1);
Index_frame = zeros(length(check_next_stack{1,ii}(1,:)),1);

    for imfeat = 1:length(check_previous_stack{1,ii-gulla}(1,:))
        for timfeat=1:length(check_next_stack{1,ii}(1,:))
            Df1(imfeat,timfeat) = pdist2(check_previous_stack{1,ii-gulla}(:,imfeat)',check_next_stack{1,ii}(:,timfeat)');
        end
    end

    for imfeat1 = 1:length(check_previous_stack{1,ii-gulla}(1,:))
        Df2(imfeat1) = pdist2(A, single(check_previous_stack{1,ii-gulla}(:,imfeat1)'));
    end

    for timfeat1 = 1:length(check_next_stack{1,ii}(1,:))
        Df3(timfeat1) = pdist2(A, single(check_next_stack{1,ii}(:,timfeat1)'));
    end

    for imfeat2 = 1:length(check_previous_stack{1,ii-gulla}(1,:))
        for timfeat2=1:length(check_next_stack{1,ii}(1,:))
            Df4(imfeat2,timfeat2) = (Df1(imfeat2,timfeat2))/ (sqrt(Df2(imfeat2)*Df3(timfeat2)));
        end
    end

    for norm = 1:length(check_next_stack{1,ii}(1,:))
        [Dist_frame(norm), Index_frame(norm)] = min(Df4(:,norm));
            if min_dist>Dist_frame(norm)
                min_dist=Dist_frame(norm);
            end
    end
    
    dist_frame_stack{ii}{gulla}=Dist_frame;
    index_frame_stack{ii}{gulla}=Index_frame;    
for poppo = 1:length(acti_next_stack_res{ii})            
    for polly = 1:length(dist_frame_stack{ii})
        if index_frame_stack{ii}{polly}
            acti_next_stack_res_dist{ii}{poppo}{polly} = dist_frame_stack{ii}{polly}(poppo);
            acti_next_stack_res_dist_min{ii}(poppo)=min(cell2mat(acti_next_stack_res_dist{ii}{poppo}(:)));
        end
    end
end
    for ret = 1:length(index_frame_stack{1,ii}{gulla})
        for rete = ret+1:length(index_frame_stack{1,ii}{gulla})
          
            if index_frame_stack{1,ii}{gulla}(ret) == index_frame_stack{1,ii}{gulla}(rete)

                if dist_frame_stack{1,ii}{gulla}(ret) < dist_frame_stack{1,ii}{gulla}(rete) && dist_frame_stack{1,ii}{gulla}(ret) < thr_fv && dist_frame_stack{1,ii}{gulla}(ret) <= acti_next_stack_res_dist_min{ii}(ret)
                    Index_min_norm_new(acti_next_stack_res{1,ii}(ret)) = Index_stack{1,ii-gulla-1}(acti_stack_res{1,ii-gulla}(index_frame_stack{1,ii}{gulla}(ret)));
                end
                
                if dist_frame_stack{1,ii}{gulla}(ret) > dist_frame_stack{1,ii}{gulla}(rete) && dist_frame_stack{1,ii}{gulla}(rete) < thr_fv && dist_frame_stack{1,ii}{gulla}(rete) <= acti_next_stack_res_dist_min{ii}(rete)
                    Index_min_norm_new(acti_next_stack_res{1,ii}(rete)) = Index_stack{1,ii-gulla-1}(acti_stack_res{1,ii-gulla}(index_frame_stack{1,ii}{gulla}(rete)));
                end
                
            else
                if dist_frame_stack{1,ii}{gulla}(ret) < thr_fv && dist_frame_stack{1,ii}{gulla}(ret) <= acti_next_stack_res_dist_min{ii}(ret)
                    Index_min_norm_new(acti_next_stack_res{1,ii}(ret)) = Index_stack{1,ii-gulla-1}(acti_stack_res{1,ii-gulla}(index_frame_stack{1,ii}{gulla}(ret)));
                end
            end    
        end
    end
end
end

%% GENERATING NEW IDS

if ii==1
    zola=Index_min_norm_new';
else
    zola=[];
    for kolla=1:10
        if (ii-kolla)> 0
            zola=cat(2,zola,Index_new_stack{ii-kolla}');
            zola=unique(zola);
        end
    end
end



DD1 = zeros(1,next_id);
DD2 = zeros(1,next_id);
DD4 = zeros(1,next_id);


for lok = 1:length(Index_min_norm)
    for loki = lok+1:length(Index_min_norm)
        if Index_min_norm_new(lok)==Index_min_norm_new(loki)
            
            if comb(Index_min_norm(lok),lok) < comb(Index_min_norm(lok),loki)
           
                for imfeat = 1:next_id 
                    DD1(imfeat) = pdist2(single(average_activation(:,imfeat)'),single(acti_stack{ii}(:,loki)'));
                end                
                for imfeat1 = 1:next_id
                    DD2(imfeat1) = pdist2(A, single(average_activation(:,imfeat1)'));
                end
                DD3 = pdist2(A, single(acti_stack{ii}(:,loki)'));
                for imfeat2 = 1:next_id
                    DD4(imfeat2) = (DD1(imfeat2))/ (sqrt(DD2(imfeat2)*DD3));
                end
                [dist_average,Index_average]=min(DD4);
                
                
                 if dist_average < thr_fv && ~ismember(Index_average,Index_min_norm_new)
                     Index_min_norm_new(loki)=Index_average;
                 else
                     
                     next_id=next_id+1;
                     Index_min_norm_new(loki)=next_id;
                     
                 end
            else
                
                for imfeat = 1:next_id
                    DD1(imfeat) = pdist2(single(average_activation(:,imfeat)'),single(acti_stack{ii}(:,lok)'));
                end                
                for imfeat1 = 1:next_id
                    DD2(imfeat1) = pdist2(A, single(average_activation(:,imfeat1)'));
                end
                DD3 = pdist2(A, single(acti_stack{ii}(:,lok)'));
                for imfeat2 = 1:next_id
                    DD4(imfeat2) = (DD1(imfeat2))/ (sqrt(DD2(imfeat2)*DD3));
                end
                [dist_average,Index_average]=min(DD4);
                
                    if dist_average < thr_fv && ~ismember(Index_average,Index_min_norm_new)
                        Index_min_norm_new(lok)=Index_average;
                        
                        
                    else
                
                        next_id=next_id+1;
                        Index_min_norm_new(lok)=next_id;
                    end
            
            end
        end
    end
end

Index_norm_stack{ii}=Index_min_norm;
%% CALCULATING AVERAGE ACTIVATIONS

if ii==1
    average_activation(:,1:14)=acti_stack_image{1};
end

if ii==1
   Index_min_norm_previous=1:14; 
   
end


%% VISUALIZATION
if ii==1
    label1 = cellstr( num2str((Index_min_norm_previous')));
else
    label1 = cellstr( num2str((Index_min_norm_previous)));
end
Index_min_norm_previous=Index_min_norm_new;
Index_new_stack{ii}=Index_min_norm_new;

%% DISPLAYING RESULTS

imshow(fullfile('C:\Users\acharyad\Documents\MATLAB\DeepLearningForComputerVision\data','town',imageNames{ii}))
hold on
text((double(cen(:,1))), double(cen(:,2)), label1, 'Color','red','FontSize',20)
pause(0.001)
hold off

clear -regexp ^res
%%
if ii==(length(imageNames)-2)
    break;
end
     
end

