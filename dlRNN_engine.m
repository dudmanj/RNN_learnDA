% ----------------------------------\\
%  ----------------------------------\\
%  Recurrent Neural Network Engine   \\
%  ----------------------------------\\
%  ----------------------------------\\

function [outputs,hidden_r,hidden_x,e,e_store] = dlRNN_engine(this_P_perturb,net,curr_input,curr_target,act_func_handle,learn_func_handle,transfer_func_handle,plotFlag)

% Initialize activation
x = zeros(net.N,1);
learn_inputs=1;

% Initialize bias units
if numel(net.biasUI)>0
    x(net.biasUI) = net.biasV;
end

e_decay = 1/500;

% Activation Function
r = act_func_handle(x);

% initialize the running average of x
x_bar = x;
e_bar = zeros(size(net.J));
i_bar = zeros(size(net.wIn));

% Initialize elegibility trace
e = zeros(size(net.J));
eI = zeros(size(net.wIn));

% Initialize network activity
niters      = size(curr_target,2);
outputs     = zeros(net.B,niters);
hidden_r    = zeros(net.N,niters);
hidden_x    = zeros(net.N,niters);
recurrent   = zeros(net.N,1);
i_input     = zeros(net.N,1);
e           = zeros(size(net.J));
de          = zeros(size(net.J));
e_store  = zeros([size(net.J) niters]);

for i = 1:niters
    
        i_input     = net.wIn*curr_input(:,i);
        
        recurrent   = net.J*r;

        % Calculate change in activation
        if max(abs(net.wFb))>0 & i>1
            excitation = -x + recurrent + i_input + net.wFb*outputs(:,i-1); % + NOISE_TERM;
        else
            excitation = -x + recurrent + i_input;
        end

        % Random fluctuation (Uniform)
        dx = 2.*(rand(net.N,1)-0.5);
        dx(this_P_perturb < rand(net.N,1)) = 0;
        
        % Sum activation changes together
        if this_P_perturb ~= -1
            x = x + net.dt_div_tau*excitation + dx;
        else
            x = x + net.dt_div_tau*excitation;
        end

        % Introduce fixed bias
        if numel(net.biasUI)>0
            x(net.biasUI) = net.biasV;
        end

        % compute rate and store previous rate
        rprev   = r;
        r       = act_func_handle(x);

        % Calculate output using supplied function
        z = r(1:net.B,1);

        % Maintain elegibility trace (only if network was perturbed)
        if this_P_perturb ~= -1
            if learn_inputs
                deltax  = learn_func_handle( x - x_bar ); 
                de      = rprev*deltax';
%                 e       = e + de;                                         % supra-linear (must maintain sign)   ORIGINAL MICONI FORMULATION             
                e       = e + de - (e_decay*e);                             % supra-linear (must maintain sign)                
            else
                deltax  = x - x_bar; 
                de      = rprev*deltax';
%                 e       = e + learn_func_handle(de);                      % supra-linear (must maintain sign)
                e       = e + learn_func_handle((de)) - (e_decay*e);          % supra-linear (must maintain sign)
            end
        end
        e_store(:,:,i) = e;
        
        % Maintain activation trace
        x_bar = net.alpha_X * x_bar + (1.0 - net.alpha_X) * x;
        i_bar =  net.alpha_X * i_bar + (1.0 - net.alpha_X) * i_input;

        outputs(1:net.B,i)    = z;
        hidden_r(:,i)         = r;
        hidden_x(:,i)         = x;

end

if plotFlag
    figure(12); clf; plot(1:niters,hidden_r(randperm(net.N,20),:)); hold on; plot(1:niters,outputs,'k','linewidth',4); drawnow; 
end
