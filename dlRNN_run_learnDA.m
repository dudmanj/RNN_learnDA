function [net_eval,net] = dlRNN_run(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,perturbLogic)

% start timer
tic
[cm]   = TNC_CreateRBColormap(1024,'rb');
[cat]   = TNC_CreateRBColormap(8,'mapb')

% parameters
max_delta_J = 1e-4;
dt          = 1;
tau         = 30;
dt_div_tau  = dt/tau;
alpha_R     = 0.33;
alpha_X     = 0.1;
eta_J       = 0.03;      % Miconi doesn't describe any testing of this param value
update      = 20;       % How frequently one should monitor learning
P_perturb   = 3./1000;  % Miconi notes that 1 and 10 Hz are also suitable     
w_var = 0.33;

% push parameters into net structure to pass through to RNN engine
net.dt_div_tau  = dt_div_tau;
net.alpha_R     = alpha_R;
net.alpha_X     = alpha_X;
net.eta_J       = eta_J;
net.update      = update;

% begin run through all input conditions
target_list = 1:length(target);
        
for cond = 1:length(target_list)
    
    curr_cond  = cond;
    curr_input  = input{curr_cond}(:,:);
    curr_target = target{curr_cond}(:,:);
    
    if perturbLogic
        % Run internal model WITH perturbations
        [outputs,hidden_r,hidden_x,e] = dlRNN_engine(0.001,net,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,1);            
    else
        % Run internal model WITHOUT perturbations
        [outputs,hidden_r,hidden_x,e] = dlRNN_engine(-1,net,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,1);    
    end
        
    % pass output through the transfer function
    outputs_t = transfer_func_handle(outputs);            
    
%     % Calculate error for target output
%     useZ    = outputs_t(:,net.eval_inds);
%     useT    = curr_target(:,net.eval_inds); 
%     err     = mean(abs(useZ(:)-useT(:)));
        
    % Calculate error as a function something like:
        cost    = 500;
%         rewTime = find( [0 diff(curr_input(2,:))]==max(curr_input(2,:)) );
        rewTime = find( [0 diff(curr_input(2,:))]>0 , 1 );
            if numel(rewTime) == 0
                rewTime = 1600;
            end
        tmp = find(outputs_t>rewTime,1);
        if numel(tmp)==1
            deltaRew = (outputs_t(tmp)-rewTime);% / numel([rewTime:numel(curr_input)]);
%             err = tmp + cost*( deltaRew ) ;
            err = cost * (  1-exp(-deltaRew/500) ) + cost * var(outputs) * w_var; % penalizing oscillatory solutions
        else
%             deltaRew = 1;
            deltaRew = numel(curr_input)-rewTime; % / numel([rewTime:numel(curr_input)]);
%             err = cost;
            err = cost + cost * var(outputs) * w_var; % penalizing oscillatory solutions
        end        
        
        
    % Save predicted error
    R_curr(curr_cond)   = err;
    R_bar(curr_cond)    = R_curr(curr_cond);

    % Compile conditions
    net_eval.cond(curr_cond).out         = outputs;
    net_eval.cond(curr_cond).e           = e;
    net_eval.cond(curr_cond).hr          = hidden_r;
    net_eval.cond(curr_cond).hx          = hidden_x;
    net_eval.err(curr_cond)              = err;
    
end

% Initialize test error with default net cumulative error
eval_error = mean(R_curr);

        figure(12); clf;
        for cond = 1:length(target_list)
            plot(1:size(net_eval.cond(1).out,2),net_eval.cond(cond).out+cond,'color',[cond/length(target_list) , 0 , 1-(cond/length(target_list))],'linewidth',2); hold on;
        end

