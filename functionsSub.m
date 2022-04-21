% Subfunctions for estimating the confirmation bias 
% Mengting Fang - 10.2021

classdef functionsSub
    methods (Static)
        %% linear regression without offset
        % S*w=e, S'*S*w=S'*e, w=inv(S'*S)*S'*e
        % inv is only defined for squre matrices
        function w = lin_regress(S, e)
        w = inv(S'*S) * S'* e;
        end
        
        %% linear regression without offset (consistent/inconsistent) - sequence level
        function [w_cons, w_incons] = lin_regress_consis(S, e, smu, c_interm_sign)
        [w_ci,] = find(c_interm_sign==sign(smu));
        w_cons = functionsSub.lin_regress(S(w_ci,:),e(w_ci));

        [w_ici,] = find(c_interm_sign~=sign(smu));
        w_incons = functionsSub.lin_regress(S(w_ici,:),e(w_ici));  
        end
        
        %% multiple linear regression without offset (consistent/inconsistent) - sample level
        function [b, b_all] = regress_consis(S, e, c_interm_sign, bPlot)
        % Choice to sample consistency
        sample_sign = sign(S);
        sign_conflict = sample_sign.*c_interm_sign;
        consistent_samples = sign_conflict;
        inconsistent_samples = sign_conflict;
        consistent_samples(consistent_samples == -1) = 0;
        inconsistent_samples(inconsistent_samples == 1) = 0;
        inconsistent_samples = -(inconsistent_samples);
        % all trials
        b = regress(e, [S.*consistent_samples S.*inconsistent_samples]);
        b_all = regress(e, S);
        
        if bPlot == 1
            hold on;
            plot(1:6, b_all(1:6), 'b-', 'LineWidth', 1.2);
            plot(7:12, b_all(7:12), 'b-', 'LineWidth', 1.2);
            plot(1:6, b(1:6), 'g-', 'LineWidth', 1.2);
            plot(7:12, b(7:12), 'g-', 'LineWidth', 1.2);
            plot(1:6, b(13:18), 'r-', 'LineWidth', 1.2);
            plot(7:12, b(19:end), 'r-', 'LineWidth', 1.2);
            plot([1 12], [0 0], 'k--');
            set(gca, 'XLim', [0 12], 'XTick', 1:12, 'YLim', [-0.1 0.25], 'FontSize', 8, 'box', 'off');
            %set(gca, 'XLim', [0 12], 'XTick', 1:12, 'YLim', [-0.1 0.32], 'FontSize', 8, 'box', 'off');
            xline(6.5, 'k--');
            axis square;offsetAxes;
        end
        end
        
        %% multiple linear regression with offset (consistent/inconsistent) - sample level (origin Bharath)
        function [b, b_all] = regress_consis_offset(S, e, c_interm_sign, bPlot)
        % Choice to sample consistency
        sample_sign = sign(S);
        sign_conflict = sample_sign.*c_interm_sign;
        consistent_samples = sign_conflict;
        inconsistent_samples = sign_conflict;
        consistent_samples(consistent_samples == -1) = 0;
        inconsistent_samples(inconsistent_samples == 1) = 0;
        inconsistent_samples = -(inconsistent_samples);
        % all trials
        b = regress(e, [ones(length(e),1) S.*consistent_samples S.*inconsistent_samples]);
        b_all = regress(e, [ones(length(e),1) S]);
        
        if bPlot == 1
            hold on;
            plot(1:6, b_all(2:7), 'b-', 'LineWidth', 1.2);
            plot(7:12, b_all(8:13), 'b-', 'LineWidth', 1.2);
            plot(1:6, b(2:7), 'g-', 'LineWidth', 1.2);
            plot(7:12, b(8:13), 'g-', 'LineWidth', 1.2);
            plot(1:6, b(14:19), 'r-', 'LineWidth', 1.2);
            plot(7:12, b(20:end), 'r-', 'LineWidth', 1.2);
            plot([1 12], [0 0], 'k--');
            set(gca, 'XLim', [0 12], 'XTick', 1:12, 'YLim', [-0.1 0.25], 'FontSize', 8, 'box', 'off');
            xline(6.5, 'k--');
            axis square;offsetAxes;
        end
        end
        
        %% Gaussian filter of memory decay
        function y_filtered = Gaussianfilter(y, r, sigma)
        GaussTemp = ones(1,r*2-1);
        for i=1:r*2-1
            GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
        end

        y_filtered = y;

        for j=r:length(y)-r
            y_filtered(j) = y(j-r+1:j+r-1)*GaussTemp';
        end
        end
        
    end
end