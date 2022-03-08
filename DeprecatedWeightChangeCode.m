

DA_trans = cumsum(TNC_CreateGaussian(700,200,1000,1)).*3;
figure(4); plot(DA_trans); hold on;

%%
activity = zeros(1,3000)
back_p          = exp(([1:numel(activity)]-numel(activity)-3)./25);
figure(900); plot(back_p); hold on
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