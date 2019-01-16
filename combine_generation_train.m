%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used for generating synthetic image pairs.
% It should be run at the path "cocoapi/MatlabAPI" of MSCOCO API.
% The MSCOCO API codes can be downloaded from "https://github.com/cocodataset/cocoapi".
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataDir='..'; prefix='instances';
dataType='train2014';
dataType_res='train2014resize_512';
labelfile_path = 'labelfiles';

transtype = 'combine';
gttype = 'gt_combine';

prop_low = 0.01;
prop_diff = 0.1;
prop_medi = 0.25;
prop_easy = 0.5;

DIFF = 1;
MEDI = 2;
EASY = 3;

new_size = 512;
save_resized_ori_img = 1;

% the shift limit
shift_up = 512;
shift_down = -512;


rotate_up = 30;
rotate_down = -30;

lumin_up = 32;
lumin_down = -32;

scale_up = 4;
scale_down = 1.1;

transf_up = 2;

max_count = 2000;

%%%%%%%%%%%%%%%%%
% The save path of generated image pairs
%%%%%%%%%%%%%%%%%
save_root = 'dataprepare/DMAC-COCO';

save_dir = sprintf('%s/%s/',save_root, dataType);
save_dir_sub = sprintf('%s%s/',save_dir,transtype);
system(['rm -rf ', save_dir_sub]);
system(['mkdir ', save_dir_sub]);
save_dir_gt = [save_dir,gttype,'/'];
system(['rm -rf ', save_dir_gt]);
system(['mkdir ', save_dir_gt]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if save_resized_ori_img == 1
    resized_img_path = sprintf('%s/%s/%s/',save_root, dataType, dataType_res);
    system(['rm -rf ', resized_img_path]);
    system(['mkdir ', resized_img_path]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% label files create
labelfile_path_fore_diff = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'fore', 'diff');
system(['rm -f',labelfile_path_fore_diff]);
fid_fore_diff= fopen(labelfile_path_fore_diff,'wt');
fprintf(fid_fore_diff,'image1,image2,label,gt1,gt2\n');
labelfile_path_back_diff = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'back', 'diff');
system(['rm -f',labelfile_path_back_diff]);
fid_back_diff = fopen(labelfile_path_back_diff,'wt');
fprintf(fid_back_diff,'image1,image2,label,gt1,gt2\n');
labelfile_path_neg_diff = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'neg', 'diff');
system(['rm -f',labelfile_path_neg_diff]);
fid_neg_diff = fopen(labelfile_path_neg_diff,'wt');
fprintf(fid_neg_diff,'image1,image2,label,gt1,gt2\n');

labelfile_path_fore_medi = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'fore', 'medi');
system(['rm -f',labelfile_path_fore_medi]);
fid_fore_medi = fopen(labelfile_path_fore_medi,'wt');
fprintf(fid_fore_medi,'image1,image2,label,gt1,gt2\n');
labelfile_path_back_medi = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'back', 'medi');
system(['rm -f',labelfile_path_back_medi]);
fid_back_medi = fopen(labelfile_path_back_medi,'wt');
fprintf(fid_back_medi,'image1,image2,label,gt1,gt2\n');
labelfile_path_neg_medi = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'neg', 'medi');
system(['rm -f',labelfile_path_neg_medi]);
fid_neg_medi = fopen(labelfile_path_neg_medi,'wt');
fprintf(fid_neg_medi,'image1,image2,label,gt1,gt2\n');

labelfile_path_fore_easy = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'fore', 'easy');
system(['rm -f',labelfile_path_fore_easy]);
fid_fore_easy = fopen(labelfile_path_fore_easy,'wt');
fprintf(fid_fore_easy,'image1,image2,label,gt1,gt2\n');
labelfile_path_back_easy = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'back', 'easy');
system(['rm -f',labelfile_path_back_easy]);
fid_back_easy = fopen(labelfile_path_back_easy,'wt');
fprintf(fid_back_easy,'image1,image2,label,gt1,gt2\n');
labelfile_path_neg_easy = sprintf('%s%s/%s_%s_%s_%s.csv', save_dir,labelfile_path, dataType, transtype, 'neg', 'easy');
system(['rm -f',labelfile_path_neg_easy]);
fid_neg_easy = fopen(labelfile_path_neg_easy,'wt');
fprintf(fid_neg_easy,'image1,image2,label,gt1,gt2\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

annFile=sprintf('%s/annotations/%s_%s.json',dataDir,prefix,dataType);
cocoGt=CocoApi(annFile);

imgIds=sort(cocoGt.getImgIds());
[m_i n] = size(imgIds);
for idx = 1 : m_i
    imgId1 = imgIds(idx);
    img1 = cocoGt.loadImgs(imgId1);
    I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img1.file_name));
    [h1, w1, c1] = size(I);
    if c1~=3
        continue;
    end
    annIds = cocoGt.getAnnIds('imgIds',imgId1);
    [annIdsa,annIdsb]=size(annIds);
    if annIdsa == 0
        disp(['Image --- ',num2str(idx),' no segmentation!']);
        continue;
    end
    I_r = imresize(I,[new_size,new_size]);
    %%%%%%%%%%%%%%%%%%%%%%%
    % save resized original image
    if save_resized_ori_img == 1
        imwrite(I_r, [resized_img_path, img1.file_name]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%
    anns = cocoGt.loadAnns(annIds);
    [crop_masks, crop_bimasks, bimasks] = masksgeneration(I, anns);
    instance_num = length(crop_masks);
    instance_idx = ceil(rand() * instance_num);
    while instance_idx < 1 || instance_idx > instance_num
        instance_idx = ceil(rand() * instance_num);
    end
    
    mask_ori_img = bimasks{instance_idx,1};
    mask_ori_img_r = imresize(mask_ori_img,[new_size,new_size]);
    prop_rate = sum(sum(mask_ori_img_r))/(new_size*new_size);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get the instance lager than prop_low
    if prop_rate < prop_low || prop_rate > prop_easy
        itr_count = 0;
        while prop_rate < prop_low || prop_rate > prop_easy
            instance_idx = ceil(rand() * instance_num);
            if instance_idx >= 1 && instance_idx <= instance_num
                mask_ori_img = bimasks{instance_idx,1};
                mask_ori_img_r = imresize(mask_ori_img,[new_size,new_size]);
                prop_rate = sum(sum(mask_ori_img_r))/(new_size*new_size);
            end
            itr_count=itr_count + 1;
            if itr_count > 100
                break;
            end
        end
        if itr_count > 100
            disp([num2str(idx),' No suitable instance!']);
            continue;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    if prop_rate < prop_diff
        prop_flag = DIFF;
    elseif prop_rate >=prop_diff && prop_rate < prop_medi
        prop_flag = MEDI;
    elseif prop_rate >=prop_medi
        prop_flag = EASY;
    end
    
    
    
    tmp_mask = cat(3,mask_ori_img_r,mask_ori_img_r,mask_ori_img_r);
    seg_ori_img_r=I_r.*tmp_mask;
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % any operation on the mask
    %% Shift
    
    seg_paste_ = seg_ori_img_r;
    mask_paste_ = mask_ori_img_r;
    
    prop_flag_p = 0;
    prop_rate_p = 0;
    count = 0;
    while prop_flag_p ~=prop_flag || prop_rate_p ~= prop_rate
        
        R = randi([shift_down,shift_up],2,1);        
        seg_paste = imtranslate(seg_paste_,[R(1),R(2)]);
        mask_paste = imtranslate(mask_paste_,[R(1),R(2)]);
        
        prop_rate_p = sum(sum(mask_paste))/(new_size*new_size);
        if prop_rate_p >= prop_low && prop_rate_p < prop_diff
            prop_flag_p = DIFF;
        elseif prop_rate_p >=prop_diff && prop_rate_p < prop_medi
            prop_flag_p = MEDI;
        elseif prop_rate_p >=prop_medi && prop_rate_p <= prop_easy
            prop_flag_p = EASY; 
        end
        
        count = count + 1;
        if count > max_count
            break;
        end
    end
    if count > max_count
        disp(['Not generate image -- ',num2str(idx)]);
        continue;
    end
    
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% Rotation
    count = 0;
    Trans_flag = rand(1);
    if Trans_flag > 0.5
        prop_flag_p = 0;
        prop_rate_p = 0;
        while prop_flag_p ~= prop_flag || prop_rate_p ~= prop_rate
            R = randi([rotate_down,rotate_up],1,1);
            
            seg_paste = imrotate(seg_paste_,R(1),'bilinear','crop');
            mask_paste = imrotate(mask_paste_,R(1),'bilinear','crop');
            prop_rate_p = sum(sum(mask_paste))/(new_size*new_size);
            if prop_rate_p >= prop_low && prop_rate_p < prop_diff
                prop_flag_p = DIFF;
            elseif prop_rate_p >=prop_diff && prop_rate_p < prop_medi
                prop_flag_p = MEDI;
            elseif prop_rate_p >=prop_medi && prop_rate_p <= prop_easy
                prop_flag_p = EASY;
            end
            
            count = count + 1;
            if count > max_count
                break;
            end
        end
    end
    if count > max_count
        disp(['Not generate image -- ',num2str(idx)]);
        continue;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% luminance
    
    Trans_flag = rand(1);
    if Trans_flag > 0.5
        prop_flag_p = 0;
        while prop_flag_p ~=prop_flag
            R = randi([lumin_down,lumin_up],1,1);
            seg_paste = seg_paste_+R(1);
            mask_paste = mask_paste_;
            prop_rate_p = sum(sum(mask_paste))/(new_size*new_size);
            if prop_rate_p >= prop_low && prop_rate_p < prop_diff
                prop_flag_p = DIFF;
            elseif prop_rate_p >=prop_diff && prop_rate_p < prop_medi
                prop_flag_p = MEDI;
            elseif prop_rate_p >=prop_medi && prop_rate_p <= prop_easy
                prop_flag_p = EASY;
            end
        end
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% scale
    Trans_flag = rand(1);
    count = 0;
    if Trans_flag > 0.5
        prop_flag_p = 0;
        stop_flag = 1;
        while prop_flag_p ~=prop_flag || stop_flag > 0
            R_flag = rand(1);
            if R_flag >= 0.5
                R_seed = rand(1);
                R = (scale_up - scale_down)*R_seed + scale_down;
            elseif R_flag < 0.5
                R_seed = rand(1);
                R = (scale_up - scale_down)*R_seed + scale_down;
                R = 1/R;
            end
            seg_paste = imresize(seg_paste_,R);
            mask_paste = imresize(mask_paste_,R);
            [m,n,c]=size(seg_paste);
            if m < new_size
                new_seg_paste = zeros(new_size,new_size,3,'uint8');
                new_mask_paste = zeros(new_size,new_size,'uint8');
                i_idx = floor((new_size - m)/2);
                new_seg_paste((i_idx+1):(i_idx+m),(i_idx+1):(i_idx+n),:)=seg_paste;
                new_mask_paste((i_idx+1):(i_idx+m),(i_idx+1):(i_idx+n))=mask_paste;
            
            elseif m > new_size
                i_idx = floor((m - new_size)/2);
                new_seg_paste = seg_paste((i_idx+1):(i_idx+new_size),(i_idx+1):(i_idx+new_size),:);
                new_mask_paste = mask_paste((i_idx+1):(i_idx+new_size),(i_idx+1):(i_idx+new_size));
            end
            stop_flag = sum(sum(mask_paste)) - sum(sum(new_mask_paste));
        
        
            seg_paste = new_seg_paste;
            mask_paste = new_mask_paste;
            prop_rate_p = sum(sum(mask_paste))/(new_size*new_size);
            if prop_rate_p >= prop_low && prop_rate_p < prop_diff
                prop_flag_p = DIFF;
            elseif prop_rate_p >=prop_diff && prop_rate_p < prop_medi
                prop_flag_p = MEDI;
            elseif prop_rate_p >=prop_medi && prop_rate_p <= prop_easy
                prop_flag_p = EASY;
                
            end
            count = count + 1;
            if count > max_count
                break;
            end
        end
    end
    if count > max_count
        disp(['Not generate image -- ',num2str(idx)]);
        continue;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% transf
    Trans_flag = rand(1);
    count = 0;
    if Trans_flag > 0.5
        prop_flag_p = 0;
        stop_flag = 1;
        while prop_flag_p ~=prop_flag || stop_flag > 0
            R_seed = rand(1);
            count=count+1;
            if count > max_count
                break;
            end
            if R_seed >= 0.5
                transf_factor = 1/(R_seed*transf_up);
                transf_edge = floor(transf_factor * new_size);
                seg_paste = imresize(seg_paste_,[transf_edge,new_size]);
                mask_paste = imresize(mask_paste_,[transf_edge,new_size]);
                new_seg_paste = zeros(new_size,new_size,3,'uint8');
                new_mask_paste = zeros(new_size,new_size,'uint8');
                
                i_idx = floor((new_size - transf_edge)/2);
                new_seg_paste((i_idx+1):(i_idx+transf_edge),:,:)=seg_paste;
                new_mask_paste((i_idx+1):(i_idx+transf_edge),:)=mask_paste;
            elseif R_seed < 0.5
                transf_factor = R_seed * (transf_up-1) + 1/transf_up;
                transf_edge = floor(transf_factor * new_size);
                seg_paste = imresize(seg_paste_,[new_size,transf_edge]);
                mask_paste = imresize(mask_paste_,[new_size,transf_edge]);
                new_seg_paste = zeros(new_size,new_size,3,'uint8');
                new_mask_paste = zeros(new_size,new_size,'uint8');
                i_idx = floor((new_size - transf_edge)/2);
                new_seg_paste(:,(i_idx+1):(i_idx+transf_edge),:)=seg_paste;
                new_mask_paste(:,(i_idx+1):(i_idx+transf_edge))=mask_paste;
            end
            
            stop_flag = sum(sum(mask_paste)) - sum(sum(new_mask_paste));
            
            seg_paste = new_seg_paste;
            mask_paste = new_mask_paste;
            prop_rate_p = sum(sum(mask_paste))/(new_size*new_size);
            if prop_rate_p >= prop_low && prop_rate_p < prop_diff
                prop_flag_p = DIFF;
            elseif prop_rate_p >=prop_diff && prop_rate_p < prop_medi
                prop_flag_p = MEDI;
            elseif prop_rate_p >=prop_medi && prop_rate_p <= prop_easy
                prop_flag_p = EASY;
                
            end
        end
    end
    if count > max_count
        continue;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%
    
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % get the second image
    c2 = 0;
    while c2 ~= 3
        id2 = ceil(rand() * m_i);
        while id2 == idx || id2 < 1 || id2 > m_i
            id2 = ceil(rand() * m_i);
        end
        imgId2 = imgIds(id2);
        img2 = cocoGt.loadImgs(imgId2);
        I2 = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img2.file_name));
        [h2, w2, c2] = size(I2);
    end
    I2_r = imresize(I2,[new_size,new_size]);
    
    mask_paste_opp = 1 - mask_paste;
    mask_paste_opp = cat(3,mask_paste_opp,mask_paste_opp,mask_paste_opp);
    composite_img = I2_r.*mask_paste_opp + seg_paste;
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The foreground label file 
    %% save label file
    %% the instance generation
    image1_len = length(img1.file_name);
    image2_len = length(img2.file_name);
    
    composed_img_name = [img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'.jpg'];
    gt1_name = [img1.file_name(1:(image1_len-4)),...
        '_', img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'.png'];
    gt2_name = [img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'_',...
        img1.file_name(1:(image1_len-4)),'.png'];
    
    %% the label file generation
    labelfile_image1 = [dataType_res, '/',...
        img1.file_name];    
    labelfile_image2 = [transtype, '/',...
        composed_img_name];
    labelfile_label = '1';
    labelfile_gt1 = [gttype,'/',gt1_name];
    labelfile_gt2 = [gttype,'/',gt2_name];
    if prop_flag == DIFF
        fid_fore = fid_fore_diff;
    elseif prop_flag == MEDI
        fid_fore = fid_fore_medi;
    elseif prop_flag == EASY
        fid_fore = fid_fore_easy;
    end
    fprintf(fid_fore,[labelfile_image1,...
        ',',labelfile_image2,...
        ',',labelfile_label,...
        ',',labelfile_gt1,...
        ',',labelfile_gt2,'\n']);
    
    
    %% the composited image and groundtruths
    mask_ori_img_r(mask_ori_img_r==1)=255;
    mask_paste(mask_paste==1)=255;
    imwrite(composite_img, [save_dir_sub, composed_img_name]);
    imwrite(mask_ori_img_r, [save_dir_gt, gt1_name]);
    imwrite(mask_paste, [save_dir_gt, gt2_name]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The background label file 
    %% save label file
    %% the instance generation  
    gt1_name = [img2.file_name(1:(image2_len-4)),...
        '_', img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'.png'];
    gt2_name = gt1_name;
    
    % label file generation
    labelfile_image1 = [dataType_res, '/',...
        img2.file_name];    
    labelfile_image2 = [transtype, '/',...
        composed_img_name];
    labelfile_label = '1';
    labelfile_gt1 = [gttype,'/',gt1_name];
    labelfile_gt2 = [gttype,'/',gt2_name];
    if prop_flag == DIFF
        fid_back = fid_back_diff;
    elseif prop_flag == MEDI
        fid_back = fid_back_medi;
    elseif prop_flag == EASY
        fid_back = fid_back_easy;
    end
    fprintf(fid_back,[labelfile_image1,...
        ',',labelfile_image2,...
        ',',labelfile_label,...
        ',',labelfile_gt1,...
        ',',labelfile_gt2,'\n']);
    
    % save the gts
    % for evaluation, convert 1 to 255. comment for real gt generation
    mask_paste_opp(mask_paste_opp==1)=255;
    %%%%
    % imwrite(composed_img, [save_dir_sub, composed_img_name]);
    imwrite(mask_paste_opp, [save_dir_gt, gt1_name]);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The negative label file 
    %% save label file
    %% the instance generation 
    labelfile_image1 = [dataType_res, '/',...
        img1.file_name];    
    labelfile_image2 = [dataType_res, '/',...
        img2.file_name];
    labelfile_label = '0';
    labelfile_gt1 = 'null';
    labelfile_gt2 = 'null';
    if prop_flag == DIFF
        fid_neg = fid_neg_diff;
    elseif prop_flag == MEDI
        fid_neg = fid_neg_medi;
    elseif prop_flag == EASY
        fid_neg = fid_neg_easy;
    end
    fprintf(fid_neg,[labelfile_image1,...
        ',',labelfile_image2,...
        ',',labelfile_label,...
        ',',labelfile_gt1,...
        ',',labelfile_gt2,'\n']);
    
    disp(['Proccessed image -- ',num2str(idx)]);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fclose(fid_fore_diff);
fclose(fid_back_diff);
fclose(fid_neg_diff);

fclose(fid_fore_medi);
fclose(fid_back_medi);
fclose(fid_neg_medi);

fclose(fid_fore_easy);
fclose(fid_back_easy);
fclose(fid_neg_easy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%