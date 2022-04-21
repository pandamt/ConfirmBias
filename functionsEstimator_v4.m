% Bayes estimators in different conditions: categorical judgement &
% attentional cue
% Mengting Fang - 04.2022

classdef functionsEstimator_v4
    methods (Static)
        %% bayes estimator in test condition: estimation with neural signal node theta
        function [e, pC_break, c_interm_pred] = bayesEstimThetaEC(test_mode, S, smu, psC, num_samples, svar, thvar, rateConsol, flagMemDecay, flagMemDecay_break, r, sigma, sigmaMotor, rateDecayC, leak_period)
        % e: estimates of generative mean
        e = zeros(length(S), 1); 
        % c_interm_pred: intermittent categorical choice predicted by the model
        c_interm_pred = zeros(length(S), 1); 
        
        % x: possible mu estimation values for evaluation
        num_x = 500; 
        S_absmax = max([abs(min(min(S))) abs(max(max(S)))]); 
        xmax = S_absmax; 
        xmin = -xmax;
        x = linspace(xmin, xmax, num_x);

        pC0 = [psC; 1-psC];  % starting category prior
        % pc_break: average posterior after the first half stimuli
        pC_break = []; 
        pmuGC0 = ones(2, num_x); % conditional prior on x 
        pmuGC0(1,:) = pmuGC0(1,:)*(1/abs(xmin)); % uniform over range
        pmuGC0(2,:) = pmuGC0(2,:)*(1/abs(xmax));
        pmuGC0(1,x>0) = 0; 
        %pmuGC0(1,x<-25) = 0; 
        pmuGC0(2,x<0) = 0;
        %pmuGC0(2,x>25) = 0;
        % pmuGC: p(mu|C) categorical stimulus prior, reflecting subjects' individual
        % expectations about the experimental distribution of stimulus orientations
        % (uniform -> adding a smooth cosine fall-off to zero)

        for k = 1:length(S)
            pC = pC0; % [0.5; 0.5]
            pmuGC = pmuGC0; % size (2, 500)

            for kk = 1:num_samples+leak_period
                if (kk<=num_samples/2) || (kk>num_samples/2+leak_period)               
                    % likelihood p(S|mu)
                    if kk<=num_samples/2
                        kk_s = kk;
                    else
                        kk_s = kk-leak_period;
                    end
                    
                    % neuarl response of S(k,kk_s): theta
                    theta = sqrt(thvar)*randn+S(k,kk_s);
                    
                    % xx: possible S estimation values for evaluation
                    num_xx = 100; 
                    xxmax = theta + 3*sqrt(thvar); % three std of the mean, 99.73%
                    xxmin = theta - 3*sqrt(thvar);
                    xx = linspace(xxmin, xxmax, num_xx);
                
                    % likelihood p(theta|S)
                    pthGS = normpdf(theta, xx, sqrt(thvar)); % size(1, 500)
                    pthGS = repmat(transpose(pthGS), 1, num_x); % size(num_xx, num_x)
                
                    % likelihood p(S|mu)
                    pSGmu = zeros(num_xx, num_x); % size(num_xx, num_x)
                    for idx = 1:num_xx
                        pSGmu(idx, :) = normpdf(xx(idx),x,sqrt(svar)); 
                    end                
                
                    pthGmu = repmat(sum(pthGS.*pSGmu, 1), 2, 1); % size (2, num_x)
                    pthGmu(1,x>0) = 0; % clip by category
                    pthGmu(2,x<0) = 0;
  
                    % 1. infer C given theta -> posterior                    
                    pCGth = sum(pthGmu.*pmuGC,2).*pC;
                    pCGth = pCGth./sum(pCGth); 
                
                    % 2. infer mu given theta and C -> posterior (marginalize over C)
                    pmuGth = sum(pthGmu.*pmuGC.*repmat(pC,1,num_x),1);
                    %pmuGth = sum(pthGmu.*pmuGC.*repmat(pC0,1,num_x),1);
                
                    if flagMemDecay(1) % memory decay on mu
                        pmuGth = functionsSub.Gaussianfilter(pmuGth, r, sigma);
                    end
        
                    pmuGth = pmuGth./trapz(x,pmuGth); % normalize
                    
                
                    % 3. update prior on x <- posterior on x (accumulation of sample
                    % evidence, but top-down weighted)
                    pmuGC = repmat(pmuGth,2,1);
                    pmuGC(1,x>0) = 0;
                    pmuGC(2,x<0) = 0;
        
                    % 4. update prior on C <- posterior on C (based on accumulated sample evidence)  
                    if flagMemDecay(2) % memory decay on C
                        pCGth = pC0+(1-rateDecayC)*(pCGth-pC0);
                    end
              
                    pC = pCGth;  
                    
                    if kk==num_samples/2
                        % average posterior after the first half stimuli
                        pC_break = [pC_break, pC]; 
                    end
                
                    if k==6
                        figure(15);
                        subplot(num_samples, 1, kk_s);
                        size(pthGmu);
                        plot(x, pmuGth);
                        hold on;
                        plot(S(k,kk_s), 0, 'ro', theta, 0, 'b*');                    
                        suptitle = sprintf('p(mu|theta): idx %d, generative mean %.2f', k, smu(k)); 
                        sgtitle(suptitle);                   
                     
                    end
                    
                else
                    % after 1st session (half of the stimuli)
                    if kk == num_samples/2 + 1 
                        
                        if pC(1)<0.5
                            c_interm_pred(k) = 1;                                      
                        else
                            c_interm_pred(k) = -1;                        
                        end                         
                        
                        % proportional consolidation
                        if strcmp(test_mode, 'proportion')
                           psC_consol = pC(1)-(pC(1)-psC)*rateConsol;
                           pCGth = [psC_consol; 1-psC_consol]; 
                           
                           pC = pCGth;
                        end
                        
                        % control 1: reset posterior on mu (uniform), but keep posterior on C
                        if strcmp(test_mode, 'control1')
                           pmuGC = pmuGC0;
                        end
                    end
                    
                    if flagMemDecay_break(1) % memory decay on mu
                        pmuGth = functionsSub.Gaussianfilter(pmuGth, r, sigma);
                    end
                    
                    pmuGth = pmuGth./trapz(x,pmuGth); % normalize
                    
                    pmuGC = repmat(pmuGth,2,1);
                    pmuGC(1,x>0) = 0;
                    pmuGC(2,x<0) = 0;                    
                    
                    if flagMemDecay_break(2) % memory decay on C
                        pCGth = pC0+(1-rateDecayC)*(pCGth-pC0);
                    end
              
                    pC = pCGth;
                    
                end
                
                if k==6
                    figure(13);
                    subplot(1, num_samples+leak_period, kk);
                    bar(pCGth);
                    set(gca,'xticklabel',{'-','+'})
                    ylim([0 1])
                    suptitle = sprintf('full p(C|theta): idx %d, generative mean %.2f', k, smu(k)); 
                    sgtitle(suptitle);
                end
            end
            % bayesian least-square (BLS) estimator
            e_raw = trapz(x, x.*pmuGth);       
            % add motor noise
            e(k) = normrnd(e_raw, sigmaMotor);
        end
        end
        
        %% bayes estimator in test condition: estimation with neural signal node theta
        function [e, pC_break, c_interm_pred] = bayesEstimThetaEA(S, smu, c_interm, psC, num_samples, svar, thvar, flagMemDecay, flagMemDecay_break, r, sigma, sigmaMotor, rateDecayC, leak_period)
        % e: estimates of generative mean
        e = zeros(length(S), 1); 
        % c_interm_pred: intermittent categorical choice predicted by the model
        c_interm_pred = zeros(length(S), 1); 
        
        % x: possible mu estimation values for evaluation
        num_x = 500; 
        S_absmax = max([abs(min(min(S))) abs(max(max(S)))]); 
        xmax = S_absmax; % + 3*sqrt(svar);
        xmin = -xmax;
        x = linspace(xmin, xmax, num_x);

        pC0 = [psC; 1-psC];  % starting category prior
        % pc_break: average posterior after the first half stimuli
        pC_break = []; 
        pmuGC0 = ones(2, num_x); % conditional prior on x 
        pmuGC0(1,:) = pmuGC0(1,:)*(1/abs(xmin)); % uniform over range
        pmuGC0(2,:) = pmuGC0(2,:)*(1/abs(xmax));
        pmuGC0(1,x>0) = 0; 
        pmuGC0(2,x<0) = 0;

        for k = 1:length(S)
            pC = pC0; % [0.5; 0.5]
            pmuGC = pmuGC0; % size (2, 500)
            
            % attentional cue
            if c_interm(k) == -1
                pC_cue = [0.75; 0.25];
            else
                pC_cue = [0.25; 0.75];
            end

            for kk = 1:num_samples+leak_period
                if (kk<=num_samples/2) || (kk>num_samples/2+leak_period)               
                    % likelihood p(S|mu)
                    if kk<=num_samples/2
                        kk_s = kk;
                    else
                        kk_s = kk-leak_period;
                    end
                    
                    % neuarl response of S(k,kk_s): theta
                    theta = sqrt(thvar)*randn+S(k,kk_s);
                    
                    % xx: possible S estimation values for evaluation
                    num_xx = 100; 
                    xxmax = theta + 3*sqrt(thvar);
                    xxmin = theta - 3*sqrt(thvar);
                    xx = linspace(xxmin, xxmax, num_xx);
                
                    % likelihood p(theta|S)
                    pthGS = normpdf(theta, xx, sqrt(thvar)); % size(1, 500)
                    pthGS = repmat(transpose(pthGS), 1, num_x); % size(num_xx, num_x)
                    % pthGS(:, 1) = normpdf(theta, xx, sqrt(thvar));
                
                    % likelihood p(S|mu)
                    pSGmu = zeros(num_xx, num_x); % size(num_xx, num_x)
                    for idx = 1:num_xx
                        pSGmu(idx, :) = normpdf(xx(idx),x,sqrt(svar)); 
                    end                
                
                    pthGmu = repmat(sum(pthGS.*pSGmu, 1), 2, 1); % size (2, num_x)
                    pthGmu(1,x>0) = 0; % clip by category
                    pthGmu(2,x<0) = 0;
  
                    % 1. infer C given theta -> posterior                   
                    pCGth = sum(pthGmu.*pmuGC,2).*pC;
                    pCGth = pCGth./sum(pCGth); 
                
                    % 2. infer mu given theta and C -> posterior (marginalize over C)
                    pmuGth = sum(pthGmu.*pmuGC.*repmat(pC,1,num_x),1);
                
                    if flagMemDecay(1) % memory decay on mu
                        pmuGth = functionsSub.Gaussianfilter(pmuGth, r, sigma);
                    end
        
                    pmuGth = pmuGth./trapz(x,pmuGth); % normalize
                    
                    if kk==num_samples/2
                        % average posterior after the first half stimuli
                        e_break(k) = trapz(x, x.*pmuGth); 
                    end
                
                    % 3. update prior on x <- posterior on x (accumulation of sample
                    % evidence, but top-down weighted)
                    pmuGC = repmat(pmuGth,2,1);
                    pmuGC(1,x>0) = 0;
                    pmuGC(2,x<0) = 0;
        
                    % 4. update prior on C <- posterior on C (based on accumulated sample evidence)  
                    if flagMemDecay(2) % memory decay on C
                        pCGth = pC0+(1-rateDecayC)*(pCGth-pC0);
                    end
              
                    pC = pCGth;  
                    
                    if kk==num_samples/2
                        % average posterior after the first half stimuli
                        pC_break = [pC_break, pC]; 
                    end
                
                    if k==6
                        figure(15);
                        subplot(num_samples, 1, kk_s);
                        size(pthGmu);
                        plot(x, pmuGth);
                        hold on;
                        plot(S(k,kk_s), 0, 'ro', theta, 0, 'b*');                    
                        suptitle = sprintf('p(mu|theta): idx %d, generative mean %.2f', k, smu(k)); 
                        sgtitle(suptitle);
                        
                    end
                    
                else
                    % after 1st session (half of the stimuli)
                    if kk == num_samples/2 + 1                        
                        if pC(1)<0.5
                            c_interm_pred(k) = 1;                                      
                        else
                            c_interm_pred(k) = -1;                        
                        end  
                        
                        pCGth = pC_cue;      
                        pC = pCGth; 
                    end
                     
                    if flagMemDecay_break(1) % memory decay on mu
                        pmuGth = functionsSub.Gaussianfilter(pmuGth, r, sigma);
                    end
                    
                    pmuGth = pmuGth./trapz(x,pmuGth); % normalize
                    
                    
                    pmuGC = repmat(pmuGth,2,1);
                    pmuGC(1,x>0) = 0;
                    pmuGC(2,x<0) = 0;
                                                                  
                    
                    if flagMemDecay_break(2) % memory decay on C
                        pCGth = pC0+(1-rateDecayC)*(pCGth-pC0);
                    end
                    
                    pC = pCGth; 
                                              
                end
                
                if k==6
                    figure(13);
                    subplot(1, num_samples+leak_period, kk);
                    bar(pCGth);
                    set(gca,'xticklabel',{'-','+'})
                    ylim([0 1])
                    suptitle = sprintf('full p(C|theta): idx %d, generative mean %.2f', k, smu(k)); 
                    sgtitle(suptitle);
                end
                
            end
            % bayesian least-square (BLS) estimator
            e_raw = trapz(x, x.*pmuGth);       
            % add motor noise
            e(k) = normrnd(e_raw, sigmaMotor);
        end
        end
    end
end