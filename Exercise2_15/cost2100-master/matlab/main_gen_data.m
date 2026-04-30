clear; clc;
rng('shuffle'); % 這邊做初始化

% 並確保輸出資料夾存在
saveDir = 'mydata';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

%% === [0] 定義所有資料集共用的基礎參數 ===
commonCfg.Network  = 'Indoor_CloselySpacedUser_2_6GHz';
commonCfg.scenario = 'LOS';
commonCfg.freq     = [2.58e9, 2.62e9];
commonCfg.Link     = 'Multiple';
commonCfg.Antenna  = 'MIMO_Cyl_patch';
commonCfg.Band     = 'Wideband';

% 在這邊給予預設 9 個使用者基礎位置
base_MSPos = [ -2.56, 1.73, 2.23; ...
               -3.08, 1.73, 2.23; ...
               -2.56, 2.62, 2.58; ...
               -4.64, 1.73, 2.23; ...
               -2.56, 4.40, 3.30; ...
               -3.08, 3.51, 2.94; ...
               -3.60, 4.40, 3.30; ...
               -4.12, 4.40, 3.30; ...
               -4.12, 2.62, 2.58 ];

%%  [1] 配置與產生 Dataset 1 
fprintf('--- Configuring and Generating Dataset 1 ---\n');
cfg1          = commonCfg;
cfg1.snapNum  = 1024;
cfg1.snapRate = 40; 
% 室內場景偏移量限制在 [-1, 1] 公尺以內
cfg1.MSPos    = generate_random_distribution(base_MSPos, 0.1, [-0.5, 0.5]);
cfg1.MSVelo   = repmat([-0.25, 0, 0], 9, 1); 
process_and_save_dataset(1, cfg1, saveDir);

%%  [2] 配置與產生 Dataset 2 
fprintf('--- Configuring and Generating Dataset 2 ---\n');
cfg2          = commonCfg;
cfg2.snapNum  = 1024;
cfg2.snapRate = 40;
cfg2.MSPos    = generate_random_distribution(base_MSPos, 0.15, [-1.5, -0.5]); % 偏左下微調
cfg2.MSVelo   = repmat([0.15, 0.1, 0], 9, 1);  
process_and_save_dataset(2, cfg2, saveDir);

%% [3] 配置與產生 Dataset 3 
fprintf('--- Configuring and Generating Dataset 3 ---\n');
cfg3          = commonCfg;
cfg3.snapNum  = 1024;       
cfg3.snapRate = 40;
cfg3.MSPos    = generate_random_distribution(base_MSPos, 0.2, [0.5, 1.5]); % 偏右上微調
cfg3.MSVelo   = repmat([0, -0.2, 0], 9, 1); 
process_and_save_dataset(3, cfg3, saveDir);

%%  [4] 配置與產生 Dataset 4 
fprintf('--- Configuring and Generating Dataset 4 ---\n');
cfg4          = commonCfg;
cfg4.snapNum  = 1024;      
cfg4.snapRate = 40;
cfg4.MSPos    = generate_random_distribution(base_MSPos, 0.3, [-1, 1]); % 較大的原地抖動
cfg4.MSVelo   = repmat([0.1, 0.15, 0], 9, 1); 
process_and_save_dataset(4, cfg4, saveDir);

%% === [5] 配置與產生 Dataset 5 ===
fprintf('--- Configuring and Generating Dataset 5 ---\n');
cfg5          = commonCfg;
cfg5.snapNum  = 1024;
cfg5.snapRate = 40;    
cfg5.MSPos    = generate_random_distribution(base_MSPos, 0.1, [0, 1.2]); 
% 在這邊給予非常微小的不規則速度
cfg5.MSVelo   = repmat([-0.1, 0.05, 0], 9, 1) + (rand(9,3) - 0.5) * 0.05;
cfg5.MSVelo(:,3) = 0; % keep Z 軸速度為0
process_and_save_dataset(5, cfg5, saveDir);

fprintf('\nAll 5 datasets have been generated successfully!\n');



function newPos = generate_random_distribution(basePos, maxJitter, offsetRange)
    % 產生加入隨機性的使用者分佈 
    [numUsers, dims] = size(basePos);
    
    % 1. 群體共同平移 (僅平移 X 和 Y，限制範圍)
    global_offset_XY = offsetRange(1) + rand(1, 2) * (offsetRange(2) - offsetRange(1));
    global_offset = [global_offset_XY, 0]; % Z 軸絕對不平移
    
    % 2. 單獨抖動 (僅抖動 X 和 Y)
    jitter = -maxJitter + rand(numUsers, dims) * (2 * maxJitter);
    jitter(:, 3) = 0; % 強制關閉 Z 軸 (高度) 的變化，為了避免脫離 COST2100 那邊的叢集設定
    
    % 3. 合成新座標
    newPos = basePos + repmat(global_offset, numUsers, 1) + jitter;
end

function process_and_save_dataset(ds_idx, cfg, saveDir)
    %  計算 BS 與 MS 相對位置 
    BSPosCenter = [0.30 -4.37 3.20]; 
    BSPosCenter = BSPosCenter - mean(cfg.MSPos);
    MSPos_centered = cfg.MSPos - repmat(mean(cfg.MSPos), size(cfg.MSPos,1), 1);
    
    BSPosSpacing = [0 0 0];
    BSPosNum = 1;
    
    %  執行 COST2100 通道模型 
    fprintf('  -> Running COST2100 model...\n');
    [~, ~, link, ~] = cost2100(cfg.Network, cfg.scenario, cfg.freq, ...
                               cfg.snapRate, cfg.snapNum, BSPosCenter, ...
                               BSPosSpacing, BSPosNum, MSPos_centered, cfg.MSVelo);
                           
    %  匯入天線與通道轉移函數 (Transfer Function) 運算 
    fprintf('  -> Calculating Channel Responses...\n');
    BSantEADF = load('BS_Cyl_EADF.mat', 'F');
    MSantPattern = load('MS_AntPattern_User.mat');
    
    delta_f = (cfg.freq(2) - cfg.freq(1)) / 256; % 256 子載波
    ir_Cyl_Patch = create_IR_Cyl_EADF(link, cfg.freq, delta_f, BSantEADF.F, MSantPattern);
    
    H_transfer = fft(ir_Cyl_Patch, [], 2); 
    
    %  CsiNet 角度-延遲域特徵轉換 
    H_user1 = squeeze(H_transfer(:, :, 1, :)); 
    H_freq_sub = H_user1(:, 1:125, 1:32); 
    H_freq_sub = permute(H_freq_sub, [1, 3, 2]); 
    
    H_padded = cat(3, H_freq_sub, zeros(cfg.snapNum, 32, 257-125));
    H_delay = ifft(H_padded, [], 3);
    H_delay = H_delay(:, :, 1:32); 
    
    H_delay = H_delay / max(abs(H_delay(:))); 
    H_real = real(H_delay) + 0.5;
    H_imag = imag(H_delay) + 0.5;
    
    HT_full = zeros(cfg.snapNum, 2, 32, 32);
    HT_full(:, 1, :, :) = H_real;
    HT_full(:, 2, :, :) = H_imag;
    HF_all = H_freq_sub; 
    
    %  切割資料集 (Train 70%, Val 15%, Test 15%) 
    fprintf('  -> Splitting and Saving files...\n');
    n_train = floor(0.7 * cfg.snapNum);
    n_val   = floor(0.15 * cfg.snapNum);
    
    train_idx = 1 : n_train;
    val_idx   = (n_train + 1) : (n_train + n_val);
    test_idx  = (n_train + n_val + 1) : cfg.snapNum;
    
    HT_train = HT_full(train_idx, :, :, :); 
    HT_val   = HT_full(val_idx,   :, :, :); 
    HT_test  = HT_full(test_idx,  :, :, :);
    HF_test  = HF_all(test_idx,   :, :);
    
    %  輸出 .mat 檔案
    HT = HT_full; 
    save(fullfile(saveDir, sprintf('DATA_Htrain_ds%d.mat', ds_idx)), 'HT', '-v7.3');
    
    HT = HT_train; 
    save(fullfile(saveDir, sprintf('DATA_Htrainin_ds%d.mat', ds_idx)), 'HT');
    
    HT = HT_val;   
    save(fullfile(saveDir, sprintf('DATA_Hvalin_ds%d.mat', ds_idx)), 'HT');
    
    HT = HT_test;  
    save(fullfile(saveDir, sprintf('DATA_Htestin_ds%d.mat', ds_idx)), 'HT');
    
    HF_all = HF_test; 
    save(fullfile(saveDir, sprintf('DATA_HtestFin_all_ds%d.mat', ds_idx)), 'HF_all');
    
    fprintf('  -> Dataset %d Saved Successfully!\n', ds_idx);
end