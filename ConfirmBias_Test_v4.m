% Basic full model for confirmation bias in test conditions
% Visualization of data performance (combined subject)
% Mengting Fang - 04.2022
                    
cond = 'attention'; % test condition: attentional cue (attention), categorical judgement (intermResp)
sub_mode = 'rtsubjects'; % subjects mode: allsubjects, bimodalsubjects, unimodalsubjects, rtsubjects
test_mode = 'proportion'; % proportion: proportional consolidation
                          % control0: no consolidation, pure leak on Mu and C during the break (intermResp)
                          % control1: reset p(mu|C) to uniform, keep p(C) (intermResp)
                          

flagMemDecay = [1,1]; % 0: no memory decay
                      % 1: memory decay 
                      % p(mu|theta) [Gaussian filter on posterior] and p(C|theta)
                      
flagMemDecay_break = [1,0]; % 0: no memory decay during break
                            % 1: memory decay during break 
                            % p(mu|theta) [Gaussian filter on posterior] and p(C|theta)
                      
rateDecayC = 0.6;  % memory decay rate of posterior p(C|theta), 0 no fogertting, 1 complete forgetting; default = 0.2  
sigma = 10; % standard deviation for Gaussian kernel
truncate = 4; % truncate the Gaussian filter at this many standard deviations
r = round(truncate*sigma); % make the radius of the filter equal to truncate standard deviations

sigmaMotor = 5; % standard deviation for Gaussian kernel for motor noise

rateConsol = 0.5; % consolidation rate in proportional condition for category judgement

num_samples = 12; % num of samples in one trial/sequence
psC = 0.5;
smin = -14;
smax = +14;
sstd = 20;
svar = sstd^2;
thstd = 3;
thvar = thstd^2;
mu = smin:1:smax;
num_mu = length(mu);
frac_pos_data = zeros(length(mu), 1);
frac_pos_pred = zeros(length(mu), 1);
leak_period = 10; % length of memory leak period during the intermediate break 

% Test Data
%data_path = sprintf('/Users/mengting/Desktop/Lab/CPC Lab/Projects/ConfirmationBias/Data/TestData/%s_estimation_%s_behavior.mat', sub_mode, cond);
data_path = sprintf('/Users/mengting/Desktop/Lab/CPC Lab/Projects/ConfirmationBias/Data/TestData/rtsubjects_estimation_%s_behavior_500ms.mat', cond);
load(data_path); % S, c_interm, estim_resp, smu

num_mu = length(mu);
frac_pos_data = zeros(length(mu), 1);
frac_pos_pred = zeros(length(mu), 1);
        
%% bayes estimator
if strcmp(cond, 'attention')
    [estim_pred, pC_break, c_interm_pred] = functionsEstimator_v4.bayesEstimThetaEA(S, smu, c_interm, psC, num_samples, svar, thvar, flagMemDecay, flagMemDecay_break, r, sigma, sigmaMotor, rateDecayC, leak_period);
else
    [estim_pred, pC_break, c_interm_pred] = functionsEstimator_v4.bayesEstimThetaEC(test_mode, S, smu, psC, num_samples, svar, thvar, rateConsol, flagMemDecay, flagMemDecay_break, r, sigma, sigmaMotor, rateDecayC, leak_period);
end      
               
% estimation error 
error_list = estim_resp - estim_pred;               
estim_error = mean(error_list.^2); % MSE
        
% average estimates for each generative mu
[smu_sorted, sort_idx] = sort(smu);
S_sorted = S(sort_idx, :);
c_interm_data_sorted = c_interm(sort_idx);
c_interm_pred_sorted = c_interm_pred(sort_idx);
smu_num = zeros(num_mu, 1);
        
estim_resp_sorted = estim_resp(sort_idx);
estim_resp_total = zeros(num_mu,1);
estim_pred_sorted = estim_pred(sort_idx);
estim_pred_total = zeros(num_mu,1);
     
estim_resp_avg = [];
estim_resp_std = [];
estim_pred_avg = [];
estim_pred_std = [];
    
mu_idx = 1;
tmp_resp_vec = [];
tmp_pred_vec = [];
for s_idx = 1:length(S)
    mu_curr = mu(mu_idx);
    if smu_sorted(s_idx) ~= mu_curr
        E_resp_vec{mu_idx} = tmp_resp_vec;
        E_pred_vec{mu_idx} = tmp_pred_vec;
        estim_resp_avg(mu_idx) = mean(tmp_resp_vec);
        estim_resp_std(mu_idx) = std(tmp_resp_vec);
        estim_pred_avg(mu_idx) = mean(tmp_pred_vec);
        estim_pred_std(mu_idx) = std(tmp_pred_vec);
        % update
        mu_idx = mu_idx+1;
        tmp_resp_vec = estim_resp_sorted(s_idx);
        tmp_pred_vec = estim_pred_sorted(s_idx);
    else                    
        if c_interm_data_sorted(s_idx) == 1
            frac_pos_data(mu_idx) = frac_pos_data(mu_idx)+1;
        end
    
        if c_interm_pred_sorted(s_idx) == 1
            frac_pos_pred(mu_idx) = frac_pos_pred(mu_idx)+1;
        end    
        
        smu_num(mu_idx) = smu_num(mu_idx)+1;        
        tmp_resp_vec = [tmp_resp_vec, estim_resp_sorted(s_idx)];
        tmp_pred_vec = [tmp_pred_vec, estim_pred_sorted(s_idx)];
        
    end                
end
        
frac_pos_data = frac_pos_data./smu_num;
frac_pos_pred = frac_pos_pred./smu_num;

E_resp_vec{mu_idx} = tmp_resp_vec;
E_pred_vec{mu_idx} = tmp_pred_vec;
estim_resp_avg(mu_idx) = mean(tmp_resp_vec);
estim_resp_std(mu_idx) = std(tmp_resp_vec);
estim_pred_avg(mu_idx) = mean(tmp_pred_vec);
estim_pred_std(mu_idx) = std(tmp_pred_vec);
    
estim_resp_avg = reshape(estim_resp_avg, 1, num_mu);
estim_resp_std = reshape(estim_resp_std, 1, num_mu);
estim_pred_avg = reshape(estim_pred_avg, 1, num_mu);
estim_pred_std = reshape(estim_pred_std, 1, num_mu);
        
error_avg = mean((estim_resp_avg-estim_pred_avg).^2);
        
error_disp = sprintf('The MSE is %.4f (average %.4f).',estim_error, error_avg);
disp(error_disp);

%% visualization
% subjects data visualization
figure(1); clf;
subplot(1,3,1);
plot(mu, frac_pos_data, 'o-', 'MarkerFaceColor', 'b');
title('Categorical Judgement');
xlabel('Stimulus orientaion (deg)');
ylabel('Fraction of positive categorical choice (+1)');
xlim([smin smax]);
ylim([0 1]);
yticks(0:0.2:1);
legend({'subjects'},'FontSize',12);

subplot(1,3,2);
plot(mu, estim_resp_std, 'b.-');
title('Subjects Estimation Standard Deviation');
xlim([-15 15]);
yticks(-15:5:15)
ylim([0 12])
yticks(0:1:12)
legend({'subjects'},'FontSize',12);

subplot(1,3,3);
plot([-20 20], [0 0], 'k--');
hold on;
plot(mu, estim_resp_avg-mu, 'b.-');
title('Subjects Mean Estimation Errors');
ylim([-5 5])
yticks(-5:1:5)
legend({'reference','subjects'},'FontSize',12);

figure(2); clf;
subplot(1,3,1);
%plot(mu, frac_pos_data, 'o', 'MarkerFaceColor', 'b');
plot(mu, frac_pos_data, 'bo-', mu, frac_pos_pred, 'rs');
title('Categorical Judgement');
xlabel('Stimulus orientaion (deg)');
ylabel('Fraction of positive categorical choice (+1)');
xlim([smin smax]);
ylim([0 1]);
yticks(0:0.2:1);
legend({'subjects','model'},'FontSize',12);


subplot(1,3,2);
plot(mu, estim_resp_std, 'b.-', mu, estim_pred_std, 'r.-');
title('Subjects and Model Estimation Standard Deviation');
ylim([0 12])
yticks(0:1:12)
legend({'subjects','model'},'FontSize',12);

subplot(1,3,3);
plot([-20 20], [0 0], 'k--');
hold on;
plot(mu, estim_resp_avg-mu, 'b.-', mu, estim_pred_avg-mu, 'r.-');
title('Subjects and Model Mean Estimation Errors');
ylim([-5 5])
yticks(-5:1:5)
legend({'reference','subjects','model'},'FontSize',12);
        
figure(3); clf;        
subplot(1,2,1);
c_interm_sign = c_interm;
[b_sub, b_all_sub] = functionsSub.regress_consis_offset(S, estim_resp, c_interm_sign, 1);
title('Subject Weights', 'FontSize', 12);

subplot(1,2,2);
if strcmp(cond, 'intermResp')
    c_interm_sign = c_interm_pred;
end
[b_model, b_all_model] = functionsSub.regress_consis_offset(S, estim_pred, c_interm_sign, 1);
title('Model Weights', 'FontSize', 12);

figure(301); clf;        
subplot(1,2,1);
c_interm_sign = c_interm;
[b_sub, b_all_sub] = functionsSub.regress_consis(S, estim_resp, c_interm_sign, 1);
title('Subject Weights', 'FontSize', 12);

subplot(1,2,2);
if strcmp(cond, 'intermResp')
    c_interm_sign = c_interm_pred;
end
[b_model, b_all_model] = functionsSub.regress_consis(S, estim_pred, c_interm_sign, 1);
title('Model Weights', 'FontSize', 12);
        
density_resp = [];
density_pred = [];
hist_resp = [];
hist_pred = [];

figure(4); clf;
for hist_idx = 1:num_mu
    subplot(7,5,hist_idx);
    hold on;
    hist_data = cell2mat(E_resp_vec(hist_idx));
    x_d = -35:0.1:35;
    density = zeros([length(x_d), 1]);
    for k = 1:length(x_d)
        density(k) = sum(normpdf(hist_data, x_d(k), 0.8));
    end
    density = density./trapz(x_d, density); % normalize
    density_resp = [density_resp, density];
    plot(x_d, density);
    xlim([-40 40]);
    xticks(-40:20:40);   
    subtitle_str = sprintf('generative mean %.2f', mu(hist_idx)); 
    title(subtitle_str);
end
suptitle = sprintf('Subjects estimates, condition: %s', cond);
sgtitle(suptitle);
        
figure(5); clf;
for hist_idx = 1:num_mu
    subplot(7,5,hist_idx);
    hold on;
    hist_data = cell2mat(E_pred_vec(hist_idx));
    x_d = -35:0.1:35;
    density = zeros([length(x_d), 1]);
    for k = 1:length(x_d)
        density(k) = sum(normpdf(hist_data, x_d(k), 0.8));
    end
    density = density./trapz(x_d, density); % normalize
    density_pred = [density_pred, density];
    plot(x_d, density);
    xlim([-40 40]);
    xticks(-40:20:40);   
    subtitle_str = sprintf('generative mean %.2f', mu(hist_idx)); 
    title(subtitle_str);
end
suptitle = sprintf('Model estimates, condition: %s', cond);
sgtitle(suptitle);

% overall distribution of estimates
figure(6); clf;
subplot(1,2,1);
im = image(density_resp,'CDataMapping','scaled');
colormap(flipud(gray));
colorbar;
caxis([0 0.12]);
xticks(1:1:length(mu));
xticklabels(string(mu));
%xtickangle(45);
yticks(1:50:length(x_d));
yticklabels(string(-35:5:35));
set(gca,'TickLength',[0.005 0.005]); % make ticks invisible 
xlabel('Stimulus orientation (deg)');
ylabel('Estimated orientation (deg)');
title_h_resp = sprintf('Subject estimates distribution, condition: %s', cond); 
title(title_h_resp);
set(gca,'YDir','normal');

subplot(1,2,2);
im = image(density_pred,'CDataMapping','scaled');
colormap(flipud(gray));
colorbar;
caxis([0 0.12]);
xticks(1:1:length(mu));
xticklabels(string(mu));
%xtickangle(45);
yticks(1:50:length(x_d));
yticklabels(string(-35:5:35));
set(gca,'TickLength',[0.005 0.005]); % make ticks invisible 
xlabel('Stimulus orientation (deg)');
ylabel('Estimated orientation (deg)');
title_h_pred = sprintf('Model estimates distribution, condition: %s', cond); 
title(title_h_pred);
set(gca,'YDir','normal');
 
figure(7); clf;
for hist_idx = 1:num_mu
    subplot(7,5,hist_idx);
    hold on;
    hist_data = cell2mat(E_resp_vec(hist_idx));
    edges = (-40:2:40);
    histogram(hist_data, edges);
    xlim([-40 40]);
    xticks(-40:20:40);            
    subtitle_str = sprintf('generative mean %.2f', mu(hist_idx)); 
    title(subtitle_str);
end
suptitle = sprintf('Subjects estimates, condition: %s', cond);
sgtitle(suptitle);
        
figure(8); clf;
for hist_idx = 1:num_mu
    subplot(7,5,hist_idx);
    hold on;
    hist_data = cell2mat(E_pred_vec(hist_idx));
    histogram(hist_data, edges);
    xlim([-40 40]);
    xticks(-40:20:40);  
    subtitle_str = sprintf('generative mean %.2f', mu(hist_idx)); 
    title(subtitle_str);
end
suptitle = sprintf('Model estimates, condition: %s', cond);
sgtitle(suptitle);

%% analysis
% unique stimulus sequences (rows)
analysis = false;

if analysis == true
conf_threshold = 0.99;
[this_S_unique, this_S_uni_m, this_S_uni_n] = unique(S, 'rows'); % this_S_unique = S(this_S_uni_m,:) and S = S_unique(this_S_uni_n,:)
data_path = sprintf('/Users/mengting/Desktop/Lab/CPC Lab/Projects/ConfirmationBias/Data/TestData/allsubjects_estimation_behavior_S_unique.mat');
load(data_path); % S_unique (232, 12)
S_conf_path = sprintf('/Users/mengting/Desktop/Lab/CPC Lab/Projects/ConfirmationBias/Data/TestData/S_unique_conf_%d.mat', conf_threshold*100);
load(S_conf_path); % S_unique_high_conf, S_unique_low_conf

[Lia_high, Locb_high] = ismember(S, S_unique_high_conf, 'rows');
[Lia_low, Locb_low] = ismember(S, S_unique_low_conf, 'rows');

all_S_idx = (1:1:length(S))';
high_conf_idx = nonzeros(all_S_idx.*Lia_high);
low_conf_idx = nonzeros(all_S_idx.*Lia_low);

S_high = S(high_conf_idx,:);
estim_resp_high = estim_resp(high_conf_idx);
estim_pred_high = estim_pred(high_conf_idx);
c_interm_high = c_interm(high_conf_idx);
c_interm_pred_high = c_interm_pred(high_conf_idx);
smu_high = smu(high_conf_idx);

S_low = S(low_conf_idx,:);
estim_resp_low = estim_resp(low_conf_idx);
estim_pred_low = estim_pred(low_conf_idx);
c_interm_low = c_interm(low_conf_idx);
c_interm_pred_low = c_interm_pred(low_conf_idx);
smu_low = smu(low_conf_idx);

figure(30); clf;
%%% high confidence
subplot(2,2,1);
c_interm_sign = c_interm_high;
[b_sub_high, b_all_sub_high] = functionsSub.regress_consis(S_high, estim_resp_high, c_interm_sign, 1);
title('Subjects Weights (High Confidence)');

subplot(2,2,2);
if strcmp(cond, 'intermResp')
    c_interm_sign = c_interm_pred_high;
end
[b_model_high, b_all_model_high] = functionsSub.regress_consis(S_high, estim_pred_high, c_interm_sign, 1);
title('Model Weights (High Confidence)');

%%% low confidence
subplot(2,2,3);
c_interm_sign = c_interm_low;
[b_sub_low, b_all_sub_low] = functionsSub.regress_consis(S_low, estim_resp_low, c_interm_sign, 1);
title('Subjects Weights (Low Confidence)');
        
subplot(2,2,4);
if strcmp(cond, 'intermResp')
    c_interm_sign = c_interm_pred_low;
end
[b_model_low, b_all_model_low] = functionsSub.regress_consis(S_low, estim_pred_low, c_interm_sign, 1);
title('Model Weights (Low Confidence)');

conf_disp = sprintf('S with high confidence: %d. S with low confidence: %d.',length(high_conf_idx), length(low_conf_idx));
disp(conf_disp);

%% Attentional cue
% high confidence in agreemnt/conflict with attentional cue
a = ones(length(S), 1);
cue_consis_idx = [];
cue_inconsis_idx = [];

for i=1:length(S)
    if c_interm_pred(i) == c_interm(i)
        cue_consis_idx = [cue_consis_idx; i];
    else
        cue_inconsis_idx = [cue_inconsis_idx; i];
    end  
end


[Lia_consis, Locb_consis] = ismember(high_conf_idx, cue_consis_idx, 'rows');
[Lia_inconsis, Locb_inconsis] = ismember(high_conf_idx, cue_inconsis_idx, 'rows');

all_S_high_conf_idx = (1:1:length(high_conf_idx))';
high_consis_cc_idx = nonzeros(all_S_high_conf_idx.*Lia_consis);
high_inconsis_cc_idx = nonzeros(all_S_high_conf_idx.*Lia_inconsis);

conf_disp = sprintf('S with consistent confidence and cue: %d. S with inconsistent confidence and cue: %d.',length(cue_consis_idx), length(cue_inconsis_idx));
disp(conf_disp);
high_conf_consis_disp = sprintf('S with high confidence and consistent cue: %d. S with high confidence and inconsistent cue: %d.',length(high_consis_cc_idx), length(high_inconsis_cc_idx));
disp(high_conf_consis_disp);

figure(32); clf;    
subplot(1,2,1); 
w = regress(estim_resp_high, S_high);
plot(w,'ks-');
hold on;
axis([1 num_samples -0.15 0.4]);
w_cons_resp = regress(estim_resp_high(high_consis_cc_idx), S_high(high_consis_cc_idx,:));
w_incons_resp = regress(estim_resp_high(high_inconsis_cc_idx), S_high(high_inconsis_cc_idx,:));  
        
plot(w_cons_resp,'bo-');
plot(w_incons_resp,'ro-');
plot([1 num_samples], [0 0], 'k--');
title('Subjects Weights (High Confidence)');
legend('across all S','consistent S','inconsistent S','reference');

subplot(1,2,2); 
w = regress(estim_pred_high, S_high);
plot(w,'ks-');
hold on;
axis([1 num_samples -0.15 0.4]);
w_cons_resp = regress(estim_pred_high(high_consis_cc_idx), S_high(high_consis_cc_idx,:));
w_incons_resp = regress(estim_pred_high(high_inconsis_cc_idx), S_high(high_inconsis_cc_idx,:));  
        
plot(w_cons_resp,'bo-');
plot(w_incons_resp,'ro-');
plot([1 num_samples], [0 0], 'k--');
title('Model Weights (High Confidence)');
legend('across all S','consistent S','inconsistent S','reference');
end
