function [net_run,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u] = dlRNN_train(net,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,tolerance,tau_trans,stim,filt_scale)
% note stim is a variable coding for lick- (-1) , no stim (0), lick+ (1)
global monitor;

%--------------- SIMULATION SCRIPT FOR MODEL IN Coddington & Dudman (2018)
            t           = 1:800;
            Mu          = 490;
            Sigma       = 20;
            tmp_gauss   = ( 1./( sqrt(2.*pi.*Sigma.*Sigma) ) ) .* exp( -(t-Mu).^2 ./ (2.*Sigma).^2 );
            integral    = trapz(tmp_gauss);
            tmp_gauss   = tmp_gauss./integral;

            da_imp_resp_f_se = tmp_gauss;

            indep_resp_func = 1;
            t           = 1:800;
            Mu          = 380;
            Sigma       = 20;
            tmp_gauss   = ( 1./( sqrt(2.*pi.*Sigma.*Sigma) ) ) .* exp( -(t-Mu).^2 ./ (2.*Sigma).^2 );
            integral    = trapz(tmp_gauss);
            tmp_gauss   = tmp_gauss./integral;

            da_imp_resp_f_ee = tmp_gauss;

            if indep_resp_func
                t       = 1:800;
                Mu      = 490;
                Sigma   = 60;
                tmp_gauss = ( 1./( sqrt(2.*pi.*Sigma.*Sigma) ) ) .* exp( -(t-Mu).^2 ./ (2.*Sigma).^2 );
                integral = trapz(tmp_gauss);
                tmp_gauss = tmp_gauss./integral;

                scale_factor = 3;
                da_imp_resp_f_ei = (da_imp_resp_f_ee.*scale_factor) - 0.9.*tmp_gauss;
                da_imp_resp_f_ei = da_imp_resp_f_ei * 1.25;
            else
                da_imp_resp_f_ei = da_imp_resp_f_ee;
            end
%--------------- SIMULATION SCRIPT FOR MODEL IN Coddington & Dudman (2018)
            
pred_da_move = [];
pred_da_sense = [];
pred_da_move_u = [];
pred_da_sense_u = [];

clear plot_stats;
pass        = 1;
stats_cnt   = 1;
[cm]        = TNC_CreateRBColormap(1024,'rb');
cost        = 500;
w_var       = 0.25;             % relative weighting of cost of variance in output unit activity
w_var_set = 0.25;
[lck_gauss] = TNC_CreateGaussian(500,25,1000,1);

error_reps  = 1;                % can execute a batch of trials to get estimate of performance
err_vector_y = 1500;

% parameters
max_delta_J = 0.01;             % prevent very large weight changes (in practice ~never invoked)
dt          = 1;
tau         = 30;
dt_div_tau  = dt/tau;
% alpha_R     = 0.9;
alpha_R     = 0.75;
alpha_X     = 0.33;
eta_J       = 5e-5;             % {1e-5 1e-4} range seems most stable for learning
% eta_J       = 5e-6;             % {1e-5 1e-4} range seems most stable for learning
eta_wIn     = 1./tau_trans;

plant_scale = 1; % moving into to plant itself (seems better; but leave this variable temporarily for future)

net_run.eta_J = eta_J;

update      = 5;                % How frequently to monitor learning
P_perturb   = 1./1000;          % Miconi notes that anhthing between 1 and 10 Hz are suitable     

% push parameters into net structure to pass through to RNN engine
net.dt_div_tau  = dt_div_tau;
net.alpha_R     = alpha_R;
net.alpha_X     = alpha_X;
net.eta_J       = eta_J;
net.update      = update;
net.P_perturb   = P_perturb;

net_out = net;
orig_J = net_out.J;

running_bar = []; running_err = []; running_ant = [];  running_lat = [];

% visualize training error:
latency_cost = cost * (1-exp(-[60:1:1500]/500)');   
emp_ant_cost = [-58.2970  528.5859  311.1233]; % empirical cost surface derived from simulations
anticip_cost =  0.33*polyval(emp_ant_cost,0:0.1:9) + ... % component that is effort-like cost
    cost * 0.36 * exp(-([0:0.1:9]-2)./1.2);    
tmptmp = (latency_cost*ones(1,numel(anticip_cost)) + (ones(numel(latency_cost),1)*anticip_cost));

% begin run through all input conditions
target_list = randperm(length(target));
        
for cond = 1:length(target_list)
    
    curr_cond   = target_list(cond);
    curr_input  = input{curr_cond};
    curr_target = target{curr_cond};
    
    % Run internal model WITHOUT perturbations
    [outputs,hidden_r,hidden_x,e,e_store] = dlRNN_engine(-1,net_out,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);    
        
            err_vector = zeros(1,error_reps);
            err_vector_x = zeros(1,error_reps);
            err_vector_y = 1500*ones(1,error_reps);
            anticip_lck = zeros(1,error_reps);
            lat_lck = zeros(1,error_reps);

        for qq=1:error_reps
            
            % pass output through the transfer function
            outputs_t = transfer_func_handle(outputs./plant_scale,filt_scale);            

            % Calculate error as a function something like:
            rewTime = find( [0 diff(curr_input(2,:))]>0 , 1 );
            tmp = find(outputs_t>rewTime,1);
            if numel(tmp)==1
                deltaRew = outputs_t(tmp)-rewTime; % / numel([rewTime:numel(curr_input)]);
            else
                deltaRew = size(curr_input,2)-rewTime; % / numel([rewTime:numel(curr_input)]);
            end
            
            err_vector(qq) = cost * (  1-exp(-deltaRew/500) ) + (cost * sum(abs(diff(outputs(1,500:1600)))) * w_var); % penalizing oscillatory solutions
            
            lat_lck(qq) = deltaRew;
            anticip_lck(qq) = numel(find(outputs_t<rewTime));
        end

            running_ant = [running_ant , mean(anticip_lck)];
            running_lat = [running_lat , median(lat_lck)];
            

            
    % Save predicted error
    err                             = mean(err_vector);
    R_curr(curr_cond)       = mean(err_vector);
    R_bar(curr_cond)        = R_curr(curr_cond);        

    % Compile conditions
            net_run.cond(curr_cond).out         = outputs;
            net_run.cond(curr_cond).e           = e;
            net_run.cond(curr_cond).hr          = hidden_r;
            net_run.cond(curr_cond).hx                = hidden_x;
            net_run.pass(pass).err(curr_cond)       = R_curr(curr_cond);
            net_run.pass(pass).chk(curr_cond).o   = outputs;
            net_run.pass(pass).anticip(curr_cond) = mean(anticip_lck);
            net_run.pass(pass).lat(curr_cond)                = median(lat_lck);
            net_run.pass(pass).sens_gain(curr_cond) = outputs(rewTime+1) - outputs(rewTime);    
    
    figure(1); 
    plot(net_run.cond(curr_cond).out); hold on;
    drawnow;
    
end

% Initialize test error with default net cumulative error
test_error = sum(R_curr);

% plot some details about internal state of the network
if monitor
    figure(12); hold off;
end


% while test_error > tolerance & pass < 2000 % stop when reward collection is very good
while pass <= 800 % stop when reward collection is very good

    % don't penalize network fluctuations until behavior is already getting pretty good
    w_var = w_var_set * (1-exp(-pass/500));
    
    for cond = 1:length(target_list)
    
        curr_cond   = target_list(cond);
        curr_input  = input{curr_cond};
        curr_target = target{curr_cond};

        % run model
        [outputs,hidden_r,hidden_x,e,e_store] = dlRNN_engine(net.P_perturb,net_out,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);    
        err_vector = zeros(1,error_reps);
        err_vector_x = zeros(1,error_reps);
        err_vector_y = 1500*ones(1,error_reps);
        anticip_lck = zeros(1,error_reps);
        lat_lck = zeros(1,error_reps);

        for qq=1:error_reps
            
            % pass output through the transfer function
            outputs_t = transfer_func_handle(outputs./plant_scale,filt_scale);            

            % Calculate error as a function something like:
            rewTime = find( [0 diff(curr_input(2,:))]>0 , 1 );
            tmp = find(outputs_t>rewTime,1);
            if numel(tmp)==1
                deltaRew = (outputs_t(tmp)-rewTime);
            else
                deltaRew = size(curr_input,2)-rewTime;
            end
            
            err = cost * (  1-exp(-deltaRew/500) ) + (cost * sum(abs(diff(outputs(1,500:1600)))) * w_var); % penalizing oscillatory solutions
            
            err_vector(qq) = err;
            err_vector_x(qq) = (cost * sum(abs(diff(outputs))) * w_var_set);
            err_vector_y(qq) = cost * (  1-exp(-deltaRew/500) );
            anticip_lck(qq) = numel(find(outputs_t<rewTime));
            lat_lck(qq) = deltaRew;

        end

        % Save predicted error
        err                                     = mean(err_vector);
        R_curr(curr_cond)                       = mean(err_vector);

        % Compile conditions
        net_run.cond(curr_cond).out             = outputs;
        net_run.cond(curr_cond).e               = e;
        net_run.cond(curr_cond).hr              = hidden_r;
        net_run.cond(curr_cond).hx              = hidden_x;
        net_run.pass(pass).err(curr_cond)       = err;
        net_run.pass(pass).chk(curr_cond).v     = outputs_t;
        net_run.pass(pass).chk(curr_cond).o     = outputs;
        net_run.pass(pass).anticip(curr_cond)   = mean(anticip_lck);
        net_run.pass(pass).lat(curr_cond)       = median(lat_lck);
        net_run.pass(pass).sens_gain(curr_cond) = outputs(rewTime+1) - outputs(rewTime);
        
%----------- Use elgibility at time of reward collection
        e = e_store(:,:,round(median(lat_lck)+rewTime));

        % Maintain original connectivity sparsity
        e = e';
        e(orig_J == 0) = 0;        
        
%----------- Computer weight updates

        % Using ~DA activity to compute updates (multiply through by derivative of policy during reward delivery component)
        eta_DA_mult = outputs(1610) - outputs(1599);
        
        % NEXT STEP: stim/lick+ or stim/lick- should alter eta_DA_mult
        switch stim
            case -1
                
                if numel(find(outputs_t>1098 & outputs_t<1598))>1
                    stim_bonus = 1;                    
                else
%                     stim_bonus = -0.25;  % this does actually work but hard to make sense of 
                    stim_bonus = 5;  % this does actually work                  
                end
                
            case 0
                
                % do nothing
                    stim_bonus = 1;                    
                
            case 1
                
                if numel(find(outputs_t>1098 & outputs_t<1598))>1
                    stim_bonus = 5;                    
                else
                    stim_bonus = 1;                    
                end
                
        end

        % Miconi-like formalism
        delta_J = -eta_J .* eta_DA_mult .* e .* (R_curr(curr_cond) - R_bar(curr_cond));
        
        % Prevent too large changes in weights
        percentClipped(curr_cond) = sum(delta_J(:) > max_delta_J | delta_J(:) < -max_delta_J) / size(delta_J,1)^2 * 100;
        delta_J(delta_J > max_delta_J) = max_delta_J;
        delta_J(delta_J < -max_delta_J) = -max_delta_J;
        delta_J(isnan(delta_J)) = 0; % just in case
        
        % Update the weight matrix
        net_out.J = net_out.J + delta_J;

%------------ Calculate the proposed weight changes at inputs
        curr_val = 1 - (1-exp(-deltaRew/500)); % current reward value normalized over {0,1} like derivative
%         pred_val_r = eta_DA_mult + stim_bonus; % predicted value at reward
        pred_val_r = outputs(1599); % predicted value at reward
        
        net_out.wIn(net.oUind,2) = net_out.wIn(net.oUind,2) + eta_wIn*(curr_val-pred_val_r)*stim_bonus;
        if net_out.wIn(net.oUind,2)>10
            net_out.wIn(net.oUind,2)=10;
        elseif net_out.wIn(net.oUind,2)<0
            net_out.wIn(net.oUind,2)=0;
        end
        
        pred_val_c = outputs(110) - outputs(99); % predicted value at cue
        net_out.wIn(net.oUind,1) = net_out.wIn(net.oUind,1) + eta_wIn*(curr_val-pred_val_r-pred_val_c)*stim_bonus;
        
        if net_out.wIn(net.oUind,1)>10
            net_out.wIn(net.oUind,1)=10;
        elseif net_out.wIn(net.oUind,1)<0
            net_out.wIn(net.oUind,1)=0;
        end
        
%------------------ Calculate the proposed weight changes at inputs

        figure(300); subplot(121); imagesc(net_out.wIn(net.oUind,:),[0 10]); subplot(122); imagesc([pred_val_c pred_val_r curr_val],[-1 1]);                

        a_delta_J(curr_cond) = median(abs(delta_J(:))); % Save magnitude of change    
        a_sum_err(curr_cond) = err; % Save magnitude of error for this condition   
       
        % output the error
        R_bar_prev(curr_cond) = R_bar(curr_cond);
        R_bar(curr_cond) = alpha_R * R_bar_prev(curr_cond) + (1.0 - alpha_R) * R_curr(curr_cond);  
                
        if monitor
            disp(['Err baselined: ' num2str(R_curr(curr_cond)-R_bar(curr_cond)) ' --- delta_J: ' num2str(median(a_delta_J)) ' --- %clipped: ' num2str(median(percentClipped))])
        end
    end


%----------------------------------------------------------------
%-------- Check overall error every {update} trials
    if mod(pass,update) == 0 || pass == 1
        
        for cond = 1:length(target_list)
            curr_cond   = target_list(cond);
            curr_input  = input{curr_cond};
            curr_target = target{curr_cond};
            
%----------------------------------------------------------------
% Run model _WITHOUT_ perturbations
            [outputs,hidden_r,hidden_x,e,e_store] = dlRNN_engine(-1,net_out,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);
            [outputs_omit,hidden_r_omit,hidden_x_omit,e_omit,e_store_omit] = dlRNN_engine(-1,net_out, input_omit{curr_cond},curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);
            [outputs_uncued,hidden_r_uncued,hidden_x_uncued,e_uncued,e_store_uncued] = dlRNN_engine(-1,net_out, input_uncued{curr_cond},curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);
        
            err_vector = zeros(1,error_reps);
            err_vector_x = zeros(1,error_reps);
            err_vector_y = 1500*ones(1,error_reps);
            anticip_lck = zeros(1,error_reps);
            lat_lck = zeros(1,error_reps);
            o_tc = []; o_ti = [];
            
            for qq=1:10
                % pass output through the transfer function
                outputs_t = transfer_func_handle(outputs./plant_scale,filt_scale);           
                outputs_t_o = transfer_func_handle(outputs_omit./plant_scale,filt_scale);           
                outputs_t_u = transfer_func_handle(outputs_uncued./plant_scale,filt_scale);           

                % Calculate error as a function something like:
                rewTime = find( [0 diff(curr_input(2,:))]>0 , 1 );
                tmp = find(outputs_t>rewTime,1);
                if numel(tmp)==1
                    deltaRew = (outputs_t(tmp)-rewTime);
                else
                    deltaRew = size(curr_input,2)-rewTime;
                end
                tmp = find(outputs_t_o>rewTime,1);
                if numel(tmp)==1
                    deltaRew_o = (outputs_t_o(tmp)-rewTime);
                else
                    deltaRew_o = size(curr_input,2)-rewTime;
                end
                tmp = find(outputs_t_u>rewTime,1);
                if numel(tmp)==1
                    deltaRew_u = (outputs_t_u(tmp)-rewTime);
                else
                    deltaRew_u = size(curr_input,2)-rewTime;
                end
                
                err = cost * (  1-exp(-deltaRew/500) ) + (cost * sum(abs(diff(outputs(1,500:1600)))) * w_var); % penalizing oscillatory solutions
                                
                err_vector(qq) = err;
                err_vector_x(qq) = (cost * sum(abs(diff(outputs))) * w_var);
                err_vector_y(qq) = cost * (  1-exp(-deltaRew/500) );
                anticip_lck(qq) = numel(find(outputs_t<rewTime));
                    anticip_lck_o(qq) = numel(find(outputs_t_o<rewTime));
                    anticip_lck_u(qq) = numel(find(outputs_t_u<rewTime));
                lat_lck(qq) = deltaRew;
                    lat_lck_o(qq) = deltaRew_o;
                    lat_lck_u(qq) = deltaRew_u;
                o_tc = [o_tc outputs_t];
                o_ti = [o_ti ones(1,numel(outputs_t))*qq];
            end

            sum_lcks = zeros(1,numel(outputs));
            sum_lcks(o_tc) = sum_lcks(o_tc)+1;
            
            if monitor
                figure(700); clf;
                hold off; 
                plot(outputs,'linewidth',2); hold on; 
                plot(outputs_uncued,'linewidth',2); 
                plot(outputs_omit,'linewidth',2); 
                plot(conv(sum_lcks,lck_gauss./max(lck_gauss),'same')./update,'k-','linewidth',2);
                plot(curr_input(1,:),'linewidth',2); plot(curr_input(2,:),'linewidth',2);
            end
            
            % Save predicted error
            R_curr(curr_cond)   = mean(err_vector);
            R_bar_prev(curr_cond) = R_bar(curr_cond);
            R_bar(curr_cond) = alpha_R * R_bar_prev(curr_cond) + (1.0 - alpha_R) * R_curr(curr_cond);  

            running_err = [running_err , [mean(err_vector_x) ; mean(err_vector_y)] ];
            running_bar = [running_bar , R_bar(curr_cond) ];
            running_ant = [running_ant , mean(anticip_lck)];
            running_lat = [running_lat , median(lat_lck)];
            
            % Compile conditions
            net_run.cond(curr_cond).out         = outputs;
            net_run.cond(curr_cond).e           = e;
            net_run.cond(curr_cond).hr          = hidden_r;
            net_run.cond(curr_cond).hx                = hidden_x;
            net_run.pass(pass).err(curr_cond)       = R_curr(curr_cond);
            net_run.pass(pass).chk(curr_cond).o   = outputs;
            net_run.pass(pass).anticip(curr_cond) = mean(anticip_lck);
                net_run.pass(pass).anticip_o(curr_cond) = mean(anticip_lck_o);
                net_run.pass(pass).anticip_u(curr_cond) = mean(anticip_lck_u);
            net_run.pass(pass).lat(curr_cond)                = median(lat_lck);
                net_run.pass(pass).lat_o(curr_cond)                = median(lat_lck_o);
                net_run.pass(pass).lat_u(curr_cond)                = median(lat_lck_u);
            net_run.pass(pass).sens_gain(curr_cond) = outputs(rewTime+1) - outputs(rewTime);
                net_run.pass(pass).sens_gain_o(curr_cond) = outputs_omit(rewTime+1) - outputs_omit(rewTime);
                net_run.pass(pass).sens_gain_u(curr_cond) = outputs_uncued(rewTime+1) - outputs_uncued(rewTime);
            
        end
        
        % Can be used to decide whether to continue learning on this condition
        % test_error      = net_run.pass(pass).err;
        test_error      = mean(err_vector_y); % pretty stringent test error looking for it to be better than 1 lick cycle for all trials.

        if monitor
            disp(['Latency error: ' num2str(test_error)])
        end
        
        plot_stats.err(stats_cnt)   = mean(net_run.pass(pass).err);

        % visualize training error
        if monitor
            [err_map] = TNC_CreateRBColormap(10,'cpb'); 
            figure(102); clf; 
            subplot(121);
            imagesc(0:0.1:9,60:1:1500,tmptmp); colormap(err_map); hold on;
            set(gca,'YDir','normal');
            hold on;
            title('Cost surface'); ylabel('Latency cost'); xlabel('Activity cost');
            hold on;
            plot(running_ant,running_lat,'-','color',[0.5 0.5 0.5],'linewidth',2); 

            subplot(122);
            plot(running_err(1,:),'r-','Linewidth',2); hold on;
            plot(running_err(2,:),'k-','Linewidth',2);
            plot(running_bar,'b-','Linewidth',1);
            
        end
        
%-------- ESTIMATE DA response using the Coddington & Dudman 2018 formalism 

        sensory_resp = act_func_handle( [0 diff(outputs)] );
        sensory_resp_o = act_func_handle( [0 diff(outputs_omit)] );
        sensory_resp_u = act_func_handle( [0 diff(outputs_uncued)] );
        pred_da_stime = sensory_resp;
        pred_da_stime(sensory_resp<0) = 0;
        pred_da_stime(1:75) = 0;
        pred_da_stime(125:1575) = 0;
        pred_da_stime(1625:3000) = 0;
        pred_da_stime_u = sensory_resp_u;
        pred_da_stime_u(sensory_resp_u<0) = 0;
        pred_da_stime_u(1:75) = 0;
        pred_da_stime_u(125:1575) = 0;
        pred_da_stime_u(1625:3000) = 0;
        
        % Find state transitions in behavior
        pred_da_time = zeros(1,size(hidden_r,2));
        for qq=1:error_reps
            [outputs_t,state] = transfer_func_handle(outputs./plant_scale,filt_scale);
            all_inits = find([0 diff(state)]==1);
            cons_inits = find(all_inits>rewTime,1);
            if numel(cons_inits)>0
                init_consume(qq) = all_inits(cons_inits);
            else
                init_consume(qq) = 0;
            end
        end
        
        if numel(find(init_consume>0))>0
            pred_da_time(round(mean(init_consume(init_consume>0)))) = (numel(find(init_consume>0)) / error_reps) / error_reps; % scale by probability of reactive init
        end

        pred_da_move = [ pred_da_move ; conv(pred_da_time,da_imp_resp_f_ei,'same') ];
        pred_da_sense = [ pred_da_sense ; conv(pred_da_stime,da_imp_resp_f_se,'same') ];

        % Find state transitions in behavior
        pred_da_time_u = zeros(1,size(hidden_r_uncued,2));
        for qq=1:error_reps
            [outputs_t_u,state_u] = transfer_func_handle(outputs_uncued,filt_scale);
            all_inits_u = find([0 diff(state_u)]==1);
            cons_inits_u = find(all_inits_u>rewTime,1);
            if numel(cons_inits_u)>0
                init_consume_u(qq) = all_inits_u(cons_inits_u);
            else
                init_consume_u(qq) = 0;
            end
        end
        
        if numel(find(init_consume_u>0))>0
            pred_da_time_u(round(mean(init_consume_u(init_consume_u>0)))) = (numel(find(init_consume_u>0)) / error_reps) / error_reps; % scale by probability of reactive init
        end

        pred_da_move_u = [ pred_da_move_u ; conv(pred_da_time_u,da_imp_resp_f_ei,'same') ];
        pred_da_sense_u = [ pred_da_sense_u ; conv(pred_da_stime_u,da_imp_resp_f_se,'same') ];

%-------- ESTIMATE DA response using the Coddington & Dudman 2018 formalism
        
        if monitor==1
            figure(103); 
            subplot(221);
            imagesc(pred_da_move);
            subplot(222);
            imagesc(pred_da_sense);
            subplot(223);
            imagesc(pred_da_move_u);
            subplot(224);
            imagesc(pred_da_sense_u);
        else            
            if mod(pass,200) == 0 & monitor==1
                disp(['Error: ' num2str(test_error) ' --- Median delta_J: ' num2str(median(a_delta_J)) ' --- %Clipped: ' num2str(median(percentClipped))])
            end
        end
        
        stats_cnt = stats_cnt+1;
        
    end
    
    pass = pass + 1;
         
end

disp(['Number of passes through training data: ' num2str(pass-1)]);
