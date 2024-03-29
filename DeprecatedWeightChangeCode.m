

DA_trans = cumsum(TNC_CreateGaussian(700,100,1000,1)).*3;
figure(4); plot(DA_trans); hold on;

%%
activity = zeros(1,3000)
back_p          = exp(([1:numel(activity)]-numel(activity)-3)./25);
figure(900); plot(back_p); hold on

%%

% load ~/'Dropbox (Personal)'/finallatdat.mat
% final_latency.data = final_latency_data;

shift = 50; % half-lick cycle shift to agree with data
figure(300); clf;
boxplot([final_latency.daaspe final_latency.deplete final_latency.nophasic final_latency.beta final_latency.data-shift]+shift,[ones(1,24) 2*ones(1,24) 3*ones(1,24) 4*ones(1,24) 5*ones(1,numel(final_latency.data))],'labels',{'DA==PE','DA deplete','DA no phasic','DA==beta','Data'});
ylabel('Final collection latency (ms)');

figure(301); clf;
bar([1 2 3 4 5],[mean(final_latency.daaspe) mean(final_latency.deplete) mean(final_latency.nophasic) mean(final_latency.beta) mean(final_latency.data)-shift]+shift);
hold on;
swarmchart([ones(1,24) 2*ones(1,24) 3*ones(1,24) 4*ones(1,24) 5*ones(1,numel(final_latency.data))],[final_latency.daaspe final_latency.deplete final_latency.nophasic final_latency.beta final_latency.data-shift]+shift,'k','filled');
ylabel('Final collection latency (ms)');

figure(302);
[p,atab,stats] = anova1(([final_latency.daaspe' final_latency.deplete' final_latency.nophasic' final_latency.beta']));
[c,m,h,nms] = multcompare(stats,'display','on','estimate','friedman')

% [p,h] = ranksum(final_latency.nobeta', final_latency.beta')

%%

% stim_list = [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 ];
% % inits = repmat(ii([141 159 166 176 191 150]),1,6);
% inits = repmat(ii([141  96 100 110 104  55]),1,6);
% inits = repmat(ii([141 159 166 176 191 150]),1,6);

% stim_list = [-1*ones(1,18) zeros(1,18) ones(1,18) 20*ones(1,18) 21*ones(1,6) 22*ones(1,6)];
% inits = repmat(ii([141 123 159 110 166 132]),1,14);
% wIn_vec = [repmat([zeros(1,6) zeros(1,6) ones(1,6)*0.33],1,4) zeros(1,12)];
% tau_vec = [repmat([ones(1,6) ones(1,6)*1.5 ones(1,6)],1,4) ones(1,12)];
% clear run;
% num_sims = 50;
% num_sims = 36;

% TUNING UP THE CUE AND REWARD LEARNING RATES
% stim_list = zeros(1,num_sims);
% inits = repmat(ii([141 123 159 110 166 132 171 180 118 100+randperm(100,9)]),1,2);
% inits = ii(100+randperm(100,num_sims));

% inits = ones(1,num_sims).*ii(141); % good
% inits = ones(1,num_sims).*ii(123); % good
% inits = ones(1,num_sims).*ii(110); % good
% inits = ones(1,num_sims).*ii(171); % pretty good
% inits = ones(1,num_sims).*ii(132); % good
% inits = ones(1,num_sims).*ii(180); % pretty good

% inits = repmat(ii([141 123 110 132 171 180]),1,6);
% wIn_vec = [rand(1,num_sims)+0.5];
% tau_vec = [rand(1,num_sims)+1];
% wIn_vec = repmat([rand(1,6).*2],1,6);
% tau_vec = [1.1:0.1:1.9];
% tmp = repmat([1:0.2:2],6,1);
% tau_vec = tmp(1:numel(tmp));
% sat_vec = repmat(randperm(6)+4,1,6);
% wIn_vec = [zeros(1,num_sims)];
% tau_vec = [ones(1,num_sims)];

% tmp = repmat([1.5 2 2.5],6,1);
% tau_vec = tmp(1:numel(tmp));
% sat_vec = ones(1,6)*10;

%%

% Using ~DA activity to compute updates (multiply through by derivative of policy during reward delivery component)
        eta_DA_mult = outputs(1610) - outputs(1599);
        PE = R_curr(curr_cond) - R_bar(curr_cond);

        % current reward value normalized over {0,1} like derivative
        curr_val = 1- (1-exp(-deltaRew/500));
        % pred_val_r = eta_DA_mult + stim_bonus; % predicted value at reward
        pred_val_r = outputs(1599); % predicted value at reward
        pred_val_c = outputs(110); % predicted value at cue
        
        error_r = curr_val-pred_val_r;
%         error_c = (error_r-pred_val_c);
%         error_c = curr_val-pred_val_c;
%         error_c = pred_val_r;
        error_c = (error_r*0.3);



        % NEXT STEP: stim/lick+ or stim/lick- should alter eta_DA_mult
        switch stim

            case -1
                if numel(find(outputs_t>1098 & outputs_t<1598))<1
                    stim_bonus = 4;                    
                else
                    stim_bonus = 1;                  
                end
                % run critic value estimator
                [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;

                
            case 0
                stim_bonus = 1;                    
                % run critic value estimator
                [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                
            case 1
                if numel(find(outputs_t>1098 & outputs_t<1598))>1
                    stim_bonus = 4;                    
                else
                    stim_bonus = 1;                    
                end
                % run critic value estimator
                [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                
            case 20
                if numel(find(outputs_t>1098 & outputs_t<1598))>1
                    stim_bonus = 4;
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,stim);
                    wIn_scaling = 1;
                else
                    stim_bonus = 1;                    
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                end

            case 21
                if numel(find(outputs_t>1098 & outputs_t<1598))>1
                    stim_bonus = 1;
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,stim);
                    wIn_scaling = 1;
                else
                    stim_bonus = 1;                    
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                end

            case 22
                if numel(find(outputs_t>1098 & outputs_t<1598))<=1
                    stim_bonus = 1;
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,stim);
                    wIn_scaling = 1;
                else
                    stim_bonus = 1;                    
                    % run critic value estimator
                    [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                end
                
            otherwise
                stim_bonus = stim;
                % run critic value estimator
                [critic] = dlRNN_criticEngine(critic,0);
                    wIn_scaling = 10;
                    
        end
        
        
        % Miconi-like formalism
%         delta_J = -eta_J .* eta_DA_mult .* e .* (R_curr(curr_cond) - R_bar(curr_cond));

        % ACTR formulation
        delta_J = -eta_J .* eta_DA_mult .* stim_bonus .* e .* PE;

        % ACTR formulation
%         delta_J = (-eta_J .* error_r .* stim_bonus .* e .* (R_curr(curr_cond) - R_bar(curr_cond)) );
        % ACTR-C formulation
%         delta_J = (-eta_J .* error_r .* stim_bonus .* e .* (R_curr(curr_cond) - R_bar(curr_cond)) ) + (eta_J * e * critic.rpe_rew);
        % ACTR-C formulation
%         delta_J = (-eta_J .* eta_DA_mult .* e .* (R_curr(curr_cond) - R_bar(curr_cond)) ) + (eta_J * e * critic.rpe_rew);
        % ACTR-C formulation
%         delta_J = (-eta_J .* (stim_bonus .* eta_DA_mult) .* e .* (R_curr(curr_cond) - R_bar(curr_cond)) ) + (eta_J * e * critic.rpe_rew);
%         delta_J = (-eta_J .* (stim_bonus .* eta_DA_mult) .* e .* error_r ) + (eta_J * e * critic.rpe_rew);
        
        % Prevent too large changes in weights
        percentClipped(curr_cond) = sum(delta_J(:) > max_delta_J | delta_J(:) < -max_delta_J) / size(delta_J,1)^2 * 100;
        delta_J(delta_J > max_delta_J) = max_delta_J;
        delta_J(delta_J < -max_delta_J) = -max_delta_J;
        delta_J(isnan(delta_J)) = 0; % just in case
        
        % Update the weight matrix
        net_out.J = net_out.J + delta_J;

%------------ Calculate the proposed weight changes at inputs
        
        % ACTR formulation
        net_out.wIn(net.oUind,2) = net_out.wIn(net.oUind,2) + (trans_sat-net_out.wIn(net.oUind,2)).*( eta_wIn .* error_r .* stim_bonus .* eta_DA_mult );
        % ACTR-C formulation
%         net_out.wIn(net.oUind,2) = net_out.wIn(net.oUind,2) + eta_wIn*error_r*stim_bonus + eta_wIn*critic.rpe_rew;
        % ACTR-C formulation
%         net_out.wIn(net.oUind,2) = net_out.wIn(net.oUind,2) + eta_wIn*error_r*stim_bonus.* eta_DA_mult + eta_wIn*critic.rpe_rew;
        % ACTR-C formulation
%         net_out.wIn(net.oUind,2) = net_out.wIn(net.oUind,2) + eta_wIn*error_r*stim_bonus + (eta_wIn/wIn_scaling)*critic.rpe_rew;
        
        if net_out.wIn(net.oUind,2)>trans_sat
            net_out.wIn(net.oUind,2)=trans_sat;
        elseif net_out.wIn(net.oUind,2)<0
            net_out.wIn(net.oUind,2)=0;
        end
        
        % ACTR formulation
        net_out.wIn(net.oUind,1) = net_out.wIn(net.oUind,1) + (trans_sat-net_out.wIn(net.oUind,1)) .* ( eta_wIn .* error_c .* stim_bonus .* eta_DA_mult  ) .* ( net_out.wIn(net.oUind,2)>2.5 );        
        % ACTR formulation
%         net_out.wIn(net.oUind,1) = net_out.wIn(net.oUind,1) + eta_wIn*error_c*stim_bonus + (eta_wIn/wIn_scaling)*critic.rpe_cue;        
        % ACTR-C formulation
%         net_out.wIn(net.oUind,1) = net_out.wIn(net.oUind,1) + eta_wIn*error_c*stim_bonus + eta_wIn*critic.rpe_cue;

        if net_out.wIn(net.oUind,1)>trans_sat
            net_out.wIn(net.oUind,1)=trans_sat;
        elseif net_out.wIn(net.oUind,1)<0
            net_out.wIn(net.oUind,1)=0;
        end
        
%------------------ Calculate the proposed weight changes at inputs