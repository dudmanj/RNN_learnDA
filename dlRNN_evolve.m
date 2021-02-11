function [test_error,export_outputs,hidden_r,lag,err_ant,ant_lck] = dlRNN_evolve(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,tolerance)

clear plot_stats;
pass        = 1;
stats_cnt   = 1;
test_error  = Inf;
[cm]        = TNC_CreateRBColormap(1024,'rb');
cost        = 500;
w_var       = 0.25;
error_reps  = 50;

% parameters
max_delta_J = 1e-4;
dt          = 1;
tau         = 30;
dt_div_tau  = dt/tau;
alpha_R     = 0.33;
alpha_X     = 0.1;
eta_J       = 0.0003;     % Miconi doesn't describe any testing of this param value
update      = 50;       % How frequently one should monitor learning
P_perturb   = 3./1000;  % Miconi notes that 1 and 10 Hz are also suitable
max_t = size(input{1},2);
plant_scale = 40;

% push parameters into net structure to pass through to RNN engine
net.dt_div_tau  = dt_div_tau;
net.alpha_R     = alpha_R;
net.alpha_X     = alpha_X;
net.eta_J       = eta_J;
net.update      = update;
net.P_perturb   = P_perturb;

% begin run through all input conditions
target_list = randperm(length(target));
export_outputs = zeros(length(target),size(target{1}(:,:),2));
        
for cond = 1:length(target_list)
    
    curr_cond   = target_list(cond);
    curr_input  = input{curr_cond}(:,:);
    curr_target = target{curr_cond}(:,:);
    
    % Run internal model WITHOUT perturbations
    [outputs,hidden_r,hidden_x,e] = dlRNN_engine(-1,net,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,0);
    
    for qq=1:error_reps
        
        % pass output through the transfer function
        outputs_t = transfer_func_handle(outputs./plant_scale,50);

        % Calculate performance error as a function something like:
        rewTime = find( [0 diff(curr_input(2,:))]>0 , 1 );
        tmp = find(outputs_t>rewTime,1);
        if numel(tmp)==1
            deltaRew = (outputs_t(tmp)-rewTime);
            opp_cost = 1;
        else
            deltaRew = numel(curr_input)-rewTime;
            opp_cost = 2;
        end

        err = opp_cost * cost * (  1-exp(-deltaRew/250) ) + (cost * sum(abs(diff(outputs(1,500:1600)))) * w_var); % penalizing oscillatory solutions

        err_vector(qq) = err;
        err_vector_ant(qq) = (cost * sum(abs(diff(outputs))) * w_var);
        lag_vector(qq) = deltaRew;

        ant_lck(qq) = numel(find(outputs_t<rewTime));        
        
    end
        
    % Save predicted error
    err                             = mean(err_vector);
    err_ant                      = mean(err_vector_ant);
    ant_lck                      = mean(ant_lck);
    R_curr(curr_cond)       = mean(err_vector);
    lag                             = mean(lag_vector);

    export_outputs(curr_cond,:) = outputs;
    
%     % Compile conditions
%     net_run.cond(curr_cond).out         = outputs;
%     net_run.cond(curr_cond).e           = e;
%     net_run.cond(curr_cond).hr          = hidden_r;
%     net_run.cond(curr_cond).hx          = hidden_x;
%     net_run.pass(pass).err(curr_cond)   = err;
%     net_run.pass(pass).err_ant(curr_cond)   = err_ant;
%     net_run.pass(pass).chk(curr_cond).o = outputs;
    
end

% Initialize test error with default net cumulative error
test_error = err;

