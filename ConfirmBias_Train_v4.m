% Basic full model for confirmation bias in training conditions
% Visualization of data performance (combined subject)
% Mengting Fang - 04.2022
                    
cond = 'estimation'; % training condition 2 & 3: estimation, estimation_intermGap

flagMemDecay = [1,1]; % 0: no memory decay
                      % 1: memory decay - Gaussian filter on posterior
                      % p(mu|theta) and p(C|theta)

flagMemDecay_break = [1,1]; % 0: no memory decay during break
                            % 1: memory decay during break 
                            % p(mu|theta) [Gaussian filter on posterior] and p(C|theta)                      
                     
rateDecayC = 0.6;  % memory decay rate of posterior p(C|theta), 0 no fogertting, 1 complete forgetting; default = 0.2  
sigma = 15; % standard deviation for Gaussian kernel
truncate = 4; % truncate the Gaussian filter at this many standard deviations
r = round(truncate*sigma); % make the radius of the filter equal to truncate standard deviations

sigmaMotor = 5; % standard deviation for Gaussian kernel for motor noise

num_samples = 12; % num of samples in one trial/sequence
psC = 0.5;
smin = -15;
smax = +15;
sstd = 20;
svar = sstd^2;
thstd = 3;
thvar = thstd^2;
num_mu = smax-smin+1;
mu = linspace(smin,smax,num_mu);
frac_pos_pred = zeros(length(mu), 1);
leak_period = 0;

% Training Data
data_path = sprintf('/Users/mengting/Desktop/Lab/CPC Lab/Projects/ConfirmationBias/Data/TrainingData/allsubjects_training_%s_behavior.mat', cond);
load(data_path); % S, estim_resp, smu
        
%% bayes estimator
[estim_pred, c_interm_pred] = functionsEstimatorTrain_v2.bayesEstimThetaE(S, smu, psC, num_samples, svar, thvar, flagMemDecay, flagMemDecay_break, r, sigma, sigmaMotor, rateDecayC, leak_period);
        
% estimation error 
error_list = estim_resp - estim_pred;               
estim_error = mean(error_list.^2); % MSE     

% average estimates for each generative mu
[smu_sorted, sort_idx] = sort(smu);
S_sorted = S(sort_idx, :);
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
        if c_interm_pred_sorted(s_idx) == 1
            frac_pos_pred(mu_idx) = frac_pos_pred(mu_idx)+1;
        end  
        
        smu_num(mu_idx) = smu_num(mu_idx)+1;        
        tmp_resp_vec = [tmp_resp_vec, estim_resp_sorted(s_idx)];
        tmp_pred_vec = [tmp_pred_vec, estim_pred_sorted(s_idx)];        
    end                
end

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
subplot(1,2,1);
plot(mu, estim_resp_std, 'b.-');
title('Subjects Estimation Standard Deviation');
xlim([-15 15]);
yticks(-15:5:15)
ylim([0 12])
yticks(0:1:12)
legend({'subjects'},'FontSize',12);

subplot(1,2,2);
plot([-20 20], [0 0], 'k--');
hold on;
plot(mu, estim_resp_avg-mu, 'b.-');
title('Subjects Mean Estimation Errors');
ylim([-5 5])
yticks(-5:1:5)
legend({'reference','subjects'},'FontSize',12);

figure(2); clf;
subplot(1,3,1);
plot(mu, frac_pos_pred, 'rs-');
title('Categorical Judgement');
xlabel('Stimulus orientaion (deg)')
ylabel('Fraction of positive categorical choice (+1)')
xlim([smin smax])
ylim([0 1])
yticks(0:0.2:1)
legend({'model'},'FontSize',12);

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
w = regress(estim_resp, S);
plot(w,'ks:');
hold on;
axis([1 num_samples -0.15 0.4]);
title('Subjects Weights');
legend('across all samples');
        
subplot(1,2,2);
w = regress(estim_pred, S);       
plot(w,'ks:');
hold on;
axis([1 num_samples -0.15 0.4]);      
title('Model Weights');
legend('across all samples');

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
xticklabels(string(-15:1:15));
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
xticklabels(string(-15:1:15));
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