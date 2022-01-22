%% RNN_dudlab_master_learnDA
% High level script to control the construction and training of RNNs
 
% Current ideas about future development:
% 1) DA transients would be estimated with licking state transition and sensory responses. Try computing these form the network
% 2) Would like to be able to modulate the learning rate proportional to DA transients
% 3) Would like to actually drive a pair of outputs that control the forward and reverse rate on licking 
% 4) Forward rate (put D1) would only have learning rate increased for positive DA transients and Reverse rate (put D2) only for decreases
% 5) Another idea is that there is a learning rate output unit trained on the network. Not sure what objective to use for this, but an interesting idea.

%% theoretical minimum error
lick_k = TNC_CreateGaussian(200,4,400,1);
% lick_k(1:200)=0;
its = 100;
l_its_max=4;
delay = zeros(l_its_max,its);
delay_u = zeros(l_its_max,its);
cost = zeros(1,its);
licks = zeros(its,3000);
figure(21); clf;
for l_its=1:l_its_max
    licks = zeros(its,3000);
    for kk=1:its

        put_policy = [zeros(1,100) ones(1,2400) zeros(1,500)].*(0*(l_its-1))+[zeros(1,100) ones(1,10).*0 zeros(1,1590) ones(1,10).*0.075.*(l_its-1).^2 zeros(1,1290)];        
        outputs_t = dlRNN_Pcheck_transfer(put_policy,1.07);
        tmp = find(outputs_t>1600,1);
        delay_u(l_its,kk) = outputs_t(tmp)-1600;

        put_policy = [zeros(1,400) ones(1,2100) zeros(1,500)].*(0.001*(l_its-1).^2)+[zeros(1,100) ones(1,10).*0.065.*(l_its-1).^2 zeros(1,1590) ones(1,10).*0.085.*(l_its-1).^2 zeros(1,1290)];        
        outputs_t = dlRNN_Pcheck_transfer(put_policy,1.07);
        tmp = find(outputs_t>1600,1);
        delay(l_its,kk) = outputs_t(tmp)-1600;
        cost(l_its,kk) = 500 * (  1-exp(-delay(kk)/500) );
        licks(its,outputs_t) = 1;
        licks(its,:) = conv(licks(its,:),lick_k,'same');

    end
%     subplot(4,1,l_its);

    subplot(2,3,1:2); plot(mean(licks)); hold on; xlabel('Time (ms)'); ylabel('Licking'); box off; legend; axis([ 0 3000 0 0.01]);
    subplot(2,3,4:5); plot(put_policy); hold on; xlabel('Time (ms)'); ylabel('Policy'); box off; axis([ 0 3000 0 0.025]);
    subplot(2,3,[3 6]); hold off; plot(1:l_its,mean(delay(1:l_its,:),2),'r'); hold on; plot(1:l_its,mean(delay_u(1:l_its,:),2),'k'); legend('Cued','Uncued'); xlabel('Learning stage'); ylabel('Latency (ms)'); box off;
    
    
end
% mean(cost)
% std(cost)
% mean(delay)

%% Calculate cost surface using the plant model
s_map = TNC_CreateRBColormap(1000,'cpb');
clear lat_mat x_vals y_vals cost_mat
% calc latencies from range of transient and sustained settings
filter1 = TNC_CreateGaussian(5000,400,10000,1);
filter1(1:5000) = 0;

t = 1:2450;
filter1 = [zeros(1,2450) [0:0.1:1] exp(-t/1000)];

sust_vec = [0:0.05:1]./3;
% trans_vec = [-5:1:5]/5;
trans_vec = [-0.1 0 10.^[-1:0.1:1]];
figure(1); clf;
        raster.y = [];
        raster.x = [];
cnt = 1;
reps=1:1000;

for curr_sust=sust_vec
    for curr_trans=trans_vec
        
        s = find(sust_vec==curr_sust);
        t = find(trans_vec==curr_trans);
        curr_polS = zeros(1,3000);
        curr_polS = [zeros(1,600) ones(1,950).*curr_sust zeros(1,1450)];
        curr_polT = zeros(1,3000);
        curr_polT(1599:1620) = curr_trans;

        activity = curr_polS + conv(curr_polT,filter1,'same');
        activity = curr_polS + curr_polT;
        
%         figure(1); plot(activity); hold on;

        lat = zeros(1,numel(reps));
        for i=reps
            [checks,state] = dlRNN_Pcheck_transfer(activity,50);
%             raster.y = [raster.y ones(1,numel(checks))*cnt];
%             raster.x = [raster.x checks];
            tmp = checks(checks>1600);
            if numel(tmp>0)
                lat(i) = tmp(1)-1600;
            else
                lat(i) = 1400;
            end
            cnt = cnt+1;
        end
        
        lat_mat(s,t) = mean(lat);
        x_vals(s,t) = curr_sust;
        y_vals(s,t) = mean(lat);
        cost_mat(s,t) = (1-exp(-lat_mat(s,t)/500));


    end
end

%         figure(2); plot(raster.x,raster.y,'.');

%
figure(100);

set(0,'DefaultFigureRenderer','painters');

% option 1 is actual surface
s = surf(sust_vec./std(sust_vec),lat_mat(1,:)*(275./1400),cost_mat');
% s = surf(sust_vec./std(sust_vec),trans_vec./std(trans_vec),cost_mat');

% option 2 is equivalent fit surface
% x = reshape(x_vals,numel(cost_mat),1);
% y = reshape(y_vals,numel(cost_mat),1);
% z = reshape(cost_mat,numel(cost_mat),1);
% plant_surf_model = fit([x y],z,'poly33');
% s = plot(plant_surf_model);

s.EdgeColor = [0.35 0.35 0.35];
s.FaceAlpha = 0.85;
s.FaceLighting = 'gouraud';
colormap(s_map);
xlabel('Sustained');
ylabel('Transient');
zlabel('Cost');
% axis([min(trans_vec) max(trans_vec) min(sust_vec) max(sust_vec) 0 max(max(cost_mat))]);
axis([0 3 0 275 0 1.2]);
view(48,30);
% set(gca,'YDir','reverse');
set(gca,'Color',[0.95 0.95 0.95]);
box on; 

%% construct the inputs & define training target 

clear;
simulation_type='pavlovian';
% simulation_type='variable_cue';
% simulation_type='variable_rew';
% simulation_type='pavlovian';

switch simulation_type
    
    case 'pavlovian'
        trans_type = 'checker'

        input = cell(1,1);
        input_uncued = cell(1,1);
        input_omit = cell(1,1);
        target = cell(1,1);
        level = 1;
        totalTime = 3000;
        
        cue1Time = zeros(1,totalTime);
        cue2Time = zeros(1,totalTime);
        cue2Time([1600 1950 2200 2450 2725 3000]) = [0.01 0.025 0.05 0.075 0.15 0.3 ];

        target1 = 1;
        target2 = -1;
        
        [cue_gauss] = TNC_CreateGaussian(500,90,1000,1);
        rect_cue_gauss = [zeros(1,500) cue_gauss(500:1000)] ./ max(cue_gauss);
        cue_gauss = cue_gauss ./ max(cue_gauss);
        
        input{1}(1, :) = zeros(1,totalTime);
        input{1}(2, :) = zeros(1,totalTime);

        input{1}(1, 100:110) = 0.7;
        input{1}(1, 600:610) = 0.3;
        input{1}(2, 1600:1610) = 1;
        input_omit{1}(1,:) =  input{1}(1,:);
        input_omit{1}(2,:) =  input{1}(2,:).*0;

        input_uncued{1}(1,:) =  input{1}(1,:).*0;
        input_uncued{1}(2,:) =  input{1}(2,:);
        
        target{1} = zeros(1, totalTime);
                
        net.eval_inds = totalTime;
        
        % plot the configuration of inputs and training targets
        figure(1); clf;
            plot(1:totalTime,input{1},'linewidth',2); axis([-1 totalTime -0.1 1.1]); box off;
            ylabel('Input');
            xlabel('Time (ms)');

    case 'blocking'

    case 'variable_rew'
               trans_type = 'checker'

        input = cell(1,10);
        target = cell(1,10);
        level = 1;
        totalTime = 3000;

        cue1Time = zeros(1,totalTime);
        cue1Time(100) = 0.1;
%         cue2Time = zeros(1,totalTime);
%         cue2Time([1600 1850 2100 2350 2600 2850]) = [0.15 0.05 0.08 0.1 0.15 0.25 ];
                
        [cue_gauss] = TNC_CreateGaussian(500,50,1000,1);
        rect_cue_gauss = [zeros(1,500) cue_gauss(500:1000)] ./ max(cue_gauss);
        cue_gauss = cue_gauss ./ max(cue_gauss);
        
        for iii=1:30
            cue2Time = zeros(1,totalTime);
            cue2Time(round(randn(1).*300)+1500) = 0.25;
            input{iii}(:, :) = zeros(3,totalTime);
            input{iii}(1, :) = conv(cue1Time,rect_cue_gauss,'same');
            input{iii}(2, :) = conv(cue2Time,rect_cue_gauss,'same');
            target{iii} = zeros(1, totalTime);
        
        end
        
        net.eval_inds = totalTime;
        
        % plot the configuration of inputs and training targets
        figure(1); clf;
        for iii=1:30
            plot(1:totalTime,input{iii}+(iii./3),'linewidth',2); hold on;
%             axis([-1 totalTime -0.1 1.1]);
        end
        box off;
        ylabel('Input');
        xlabel('Time (ms)');
        
    case 'variable_cue'
        trans_type = 'checker'

        input = cell(1,10);
        target = cell(1,10);
        level = 1;
        totalTime = 3000;

        cue1Time = zeros(1,totalTime);
        cue1Time(100) = 0.1;
        cue2Time = zeros(1,totalTime);
        cue2Time([1600 1850 2100 2350 2600 2850]) = [0.15 0.05 0.08 0.1 0.15 0.25 ];
        cue1Time(100) = 0.1;
                
        [cue_gauss] = TNC_CreateGaussian(500,50,1000,1);
        rect_cue_gauss = [zeros(1,500) cue_gauss(500:1000)] ./ max(cue_gauss);
        cue_gauss = cue_gauss ./ max(cue_gauss);
        
        for iii=1:10
            cue3Time = zeros(1,totalTime);
            cue3Time(randperm(2900,1)+50) = 0.1;
            input{iii}(:, :) = zeros(3,totalTime);

            input{iii}(1, :) = conv(cue1Time,rect_cue_gauss,'same');
            input{iii}(2, :) = conv(cue2Time,cue_gauss,'same');
                input{iii}(2, 1000:1600) = 0;
            input{iii}(3, :) = conv(cue3Time,rect_cue_gauss,'same');

            target{iii} = zeros(1, totalTime);
        
        end
        
        net.eval_inds = totalTime;
        
        % plot the configuration of inputs and training targets
        figure(1); clf;
        for iii=1:10
            plot(1:totalTime,input{iii}+(iii./3),'linewidth',2); hold on;
%             axis([-1 totalTime -0.1 1.1]);
        end
        box off;
        ylabel('Input');
        xlabel('Time (ms)');
            
    otherwise
        disp('unsupported simultaion type');
        return;
        
end

% return: input, target

%% parameterize the RNN

% actFunType = 'linear' 
% actLogic = 1;
actFunType = 'tanh' 
global monitor;
actLogic = 0;
    lognorm    = 1;     % Use log normal distribution input weights


% Reasonable default values
N   = 50;              % RNN Units
B   = size(target{1},1);% Outputs
I   = size(input{1},1); % Inputs
p   = 0.9;                % Sparsity
% p   = 0.2;                % Sparsity
% g   = 1;            % Spectral scaling
% g   = 3;              % Sub-spectral scaling
g   = 1.3;              % Sub-spectral scaling
% g   = 0.75;           % Spectral scaling
% g   = 0.4;            % Spectral scaling
dt  = 1;                % Time step
% tau = 20;             % Time constant
tau = 25;               % Time constant
% tau = 50;               % Time constant
fb  = 0;                % Feedback (logic) from output units
biasUnitCnt = 0;        % number of fixed bias units in RNN
[cm] = TNC_CreateRBColormap(1024,'rb'); 

% Initialize inputs and input weights

if  strmatch(simulation_type, 'gain')
    net.wIn(:,1)    = [randn(N./2,1) ; zeros(N./2,1)] ;
    net.wIn(:,2)    = [zeros(N./2,1) ; randn(N./2,1)] ;
    net.wIn(:,3)    = [rand(N./2,1) ; zeros(N./2,1)] ;
    net.wIn(:,4)    = [zeros(N./2,1) ; rand(N./2,1)] ;
elseif strmatch(simulation_type, 'select')
    net.wIn(:,1)    = [randn(N./2,1) ; zeros(N./2,1)] ;
    net.wIn(:,2)    = [zeros(N./2,1) ; randn(N./2,1)] ;
    net.wIn(:,3)    = 2*(rand(N,1)-0.5); % range from -1 to 1    
else
    if lognorm==1
        net.wIn         = lognrnd(-2,0.6,[N,I]);
    else
        net.wIn         = ((rand(N,I)-0.5)); % range from -1 to 1
    end
end
[wIn_v,wIn_i]   = sort(mean(net.wIn,2));

%% construct the recurrent network

% Connectivity is normally distributed, scaled by the size of the network,
% the sparity, and spectral scaling factor, g.
net.N   = N;
net.p   = p;
net.g   = g;
net.dt  = dt;
net.tau = tau;
net.P_perturb = 0.003;
net.alpha_cost = 0.0;

J = zeros(N,N);
for i = 1:N
    for j = 1:N
        if rand <= p
            J(i,j) = g * randn / sqrt(p*N);
        end
    end
end

% if actLogic
%     tmp = find(J(B,:)<0);
%     J(B,tmp)=0;
% end
net.J   = J;

% Initialize output units
net.B       = B;
net.oUind   = 1:B;

% visualize the weight matrix and output units
figure(2); subplot(5,1,1);
    imagesc(net.wIn',[-1 1]); colormap(cm); ylabel('Inputs');
figure(2); subplot(5,1,2:4);
    imagesc(net.J(wIn_i,:),[-1 1]); colormap(cm); ylabel('Recurrent weights');
figure(2); subplot(5,1,5);
    imagesc(net.J(net.oUind,:),[-1 1]); colormap(cm); ylabel('Outputs');

% set up feedback if desired
if fb
    net.wFb = 2*(rand(N,B)-0.5); % range from -1 to 1
else
    net.wFb = zeros(N,B);    
end

% bias inputs
if biasUnitCnt>0
    net.biasUI  = [B+1:B+biasUnitCnt];
    net.biasV   = 2*(rand(1,biasUnitCnt)-0.5);
else
    net.biasUI  = [];
    net.biasV   = [];
end
% noise parameters?

% make sure require net to amplify sensory inputs
net.wIn(net.oUind,:) = 0;

% return: net

%% create activation function (e.g. tanh)

switch actFunType
    case 'tanh'
        act_func_handle = @tanh;
    case 'linear'
        act_func_handle = @relu;
    otherwise
        disp('bas choice'); 
        return;
end

% return: act_func_handle

%% create learning function (e.g. Miconi rule)
learn_type = 'squared'
switch learn_type
    
    case 'cubed'
        learn_func_handle = @(x) x.^3; % FYI: this shit is slooooooow (cubing 40,0000 values)
        
    case 'squared'
        learn_func_handle = @(x) x.*abs(x); % generally use this first, mucho faster operation on NxN matrix
        
    case 'mesh'

    otherwise
        disp('Unsupported learning function type.');

end

% return: learn_func_handle

%% create tranfer function (e.g. a mapping from read out neuron firing rate to a physical variable)

switch trans_type
    case 'pass'
        transfer_func_handle = @(x) x;
        
    case 'checker'
        transfer_func_handle = @dlRNN_Pcheck_transfer;
        
    otherwise
        disp('Unsupported transfer function type.');
end


% return: transfer_func_handle

%% evolve the RNN initializations

generations = 500;
[cm]        = TNC_CreateRBColormap(generations,'mbr');
evol.err = zeros(1,generations);
evol.lag = zeros(1,generations);
evol.g = zeros(1,generations);
evol.tau = zeros(1,generations);
evol.out = zeros(generations,size(input{1},2));

% N_vec = [50 100 150 200 250];
% g_vec = [ 0.5 0.8 1 1.3 1.5 2 3];
% g_vec = [0.5 1 1.3 1.5];
g_vec = 1.3;
% net.p = 1;
% tau_vec = [5 10 20 25 50];
% tau_vec = [20];
% p_vec = [0.25 0.5 0.75 1];
p_vec = 0.9;

for jj=1:generations
    
%     net.N   = N_vec(randperm(5,1));
    net.g   = g_vec(randperm(numel(g_vec),1));
%     net.tau = tau_vec(randperm(numel(tau_vec),1));    
    net.p = p_vec(randperm(numel(p_vec),1));    
    
    J = zeros(net.N,net.N);
    for i = 1:net.N
        for j = 1:net.N
            if rand <= net.p
                J(i,j) = net.g * randn / sqrt(net.p*net.N);
            end
        end
    end
    net.J   = J;


    % Initialize output units
    net.B       = B;
    net.oUind   = 1:B;

    % make sure require net to amplify sensory inputs
%     net.wIn(net.oUind,1) = 0.5;
%     net.wIn(net.oUind,2) = 1;
    net.wIn(net.oUind,1) = 0;
    net.wIn(net.oUind,2) = 0;


    % set up feedback if desired
    if fb
        net.wFb = 2*(rand(N,B)-0.5); % range from -1 to 1
    else
        net.wFb = zeros(N,B);    
    end

    % bias inputs
    if biasUnitCnt>0
        net.biasUI  = [B+1:B+biasUnitCnt+1];
        net.biasV   = 2*(rand(1,biasUnitCnt)-0.5);
    else
        net.biasUI  = [];
        net.biasV   = [];
    end
    
    [test_error,export_outputs,hidden_r,lag,err_ant,ant_lck] = dlRNN_evolve(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,50);

    evol.err(jj) = test_error;
    evol.err_ant(jj) = err_ant;
    evol.ant_lck(jj) = ant_lck;
    evol.eig(jj) = eigs(net.J,1);
    evol.lag(jj) = lag;
    all_out.gen(jj).out = export_outputs;
    evol.out(jj,:) = mean(export_outputs,1);
    evol.g(jj) = net.g;
    evol.tau(jj) = net.tau;
    evol.p(jj) = net.p;
    gens.dets(jj).net = net;
    gens.dets(jj).err = test_error;
    gens.dets(jj).out = mean(export_outputs,1);

end

%% Plot output of the evolved networks
%----------------------------------------------
 
% WOULD BE REALLY INTERESTING TO INVESTIGATE WHAT MAKES THESE NETWORKS WORK
%Perhaps try to take difference in leading eigenvalues of weight matrices and compare to
% difference in errors, or distance between matrices, or norms of the
% matrix, etc. Seems like only the eigenvalues should matter, but I do
% wonder.

[bb,ii]=sort(evol.err);
[bbb,iii]=sort(evol.lag);

figure(9); subplot(1,6,1:2); imagesc(evol.out(ii,:),[-1 1]); colormap(cm); ylabel('Generations');  xlabel('Time'); box off;
subplot(1,6,3); plot(evol.lag(ii),1:generations,'k.'); set(gca,'YDir','reverse'); box off; xlabel('Collect latency');
subplot(1,6,4); plot(evol.p(ii),1:generations,'k.'); set(gca,'YDir','reverse'); box off; xlabel('Sparsity');
subplot(1,6,5); semilogx(evol.g(ii),1:generations,'k.'); set(gca,'YDir','reverse'); box off;  xlabel('g');
subplot(1,6,6); plot(abs(evol.eig(ii)),1:generations,'k.'); set(gca,'YDir','reverse'); box off;  xlabel('eig'); set(gca,'TickDir','out');


figure(8); clf;
subplot(1,3,1); plot(evol.g,evol.err,'k+'); box off; xlabel('g'); ylabel('error');
subplot(1,3,2); plot(evol.p,evol.err,'k+'); box off; xlabel('p'); ylabel('error');
subplot(1,3,3); plot(evol.eig.^2,evol.err,'k+'); box off; xlabel('eig'); ylabel('error');
% subplot(2,2,4); semilogx(evol.tau,evol.err,'k+'); box off; xlabel('tau'); ylabel('error'); 

%----------------------------------------------
% Find median initialized network
%----------------------------------------------

% take best 500 ms tau network
% index = find(tmp > median(tmp(tmp_g==2)),1);
index = ii(1);
net = gens.dets(index).net;
figure(10); clf; subplot(211); plot(gens.dets(index).out); title('Optimal evolved '); box off;ylabel('Output unit'); axis([0 numel(gens.dets(1).out) -1 1]);
gens.dets(index).err

% take best 15 ms tau network
% index = find(tmp > median(tmp),1);
% index = find(tmp == min(tmp(tmp_tau==2000)),1);
% index = find(tmp == min(tmp(tmp_g==1.3)),1);
% index=ii(50);
% index = find(tmp > 750 & tmp < 1000 & tmpL > 1000 & tmpL < 1500,1);

figure(11); clf; plot(eig(gens.dets(ii(1)).net.J),'o'); hold on; plot(eig(gens.dets(index).net.J),'o');


figure(12); clf; % plot a bunch of example outputs
indices = [ii(1) ii(randperm(500,10))];
[cost_map] = TNC_CreateRBColormap(max(evol.lag),'wblue');

for index=indices
    net = gens.dets(index).net;
    if index==indices(1)
        plot(gens.dets(index).out,'linewidth',3,'color',cost_map(round(evol.lag(index)-min(evol.lag(indices)))+1,:)); hold on;
    else
        plot(gens.dets(index).out,'linewidth',1,'color',cost_map(round(evol.lag(index)-min(evol.lag(indices)))+1,:)); hold on;
    end
end
title('Optimal evolved '); box off; ylabel('Output unit'); axis([0 numel(gens.dets(1).out) -1 1]);

% index=ii(212)
% index=ii(101)
% index=ii(102)
% index=ii(103)
index=ii(159)

% index=ii(round(generations/2))
gens.dets(index).net.g

% index = ii(round(generations./5));
net = gens.dets(index).net;
net_init = gens.dets(index).net;
% gens.dets(index).err
figure(10); subplot(212); plot(gens.dets(index).out); title('Median evolved ');  xlabel('Time'); ylabel('Output unit'); box off; axis([0 numel(gens.dets(1).out) -1 1]);

% net.wIn(net.oUind,:) = 0;

%% Compute the empirical anticipatory licking cost function

emp_ant_cost = polyfit(evol.ant_lck*10,evol.err_ant,3) 

bin_dat = TNC_BinAndMean(evol.ant_lck*10,evol.err_ant,7);

figure(2); clf; 
plot(evol.ant_lck*10,evol.err_ant,'k.','color',[0.5 0.5 0.5 0.5]);
hold on;
plot(bin_dat.bins.center,bin_dat.bins.avg,'ko','MarkerSize',10,'MarkerFace','k');
plot(0:0.1:9,polyval(emp_ant_cost,0:0.1:9),'r-');

%% train the RNN
stim_list = [-1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1];
% stim_list = [stim_list stim_list stim_list];
% stim_list = zeros(1,36);
% stim_list = [zeros(1,9) ones(1,9)*2];

% inits = repmat([ii(101) ii(102) ii(103)],1,9);

% inits = repmat([ii(156) ii(146) ii(150)],1,6);

% inits = repmat(ii([156 150 189 173 197 210 218 209 164]),1,2);
% inits = repmat(ii([156 150 189]),1,6);
inits = repmat(ii([159 166 191 195 171 158]),1,6);

switch simulation_type
   
    case 'pavlovian'

        clear run;
        figure(1); clf;
        parfor g = 1:numel(stim_list)

            net_init = gens.dets(inits(g)).net % diverse initial states
            
%             net_init.wIn(net.oUind,:) = [0 0];
%             tau_trans = 1; % now controls wJ learning rate
            
            % check Fig 3 equivalent
            if g<=numel(stim_list)/2
                net_init.wIn(net.oUind,:) = [0 0];
                tau_trans = 1; % now controls wJ learning rate
            else
                net_init.wIn = net_init.wIn*3;
%                 net_init.wIn(net.oUind,:) = [0 2];
                net_init.wIn(net.oUind,:) = [0 0];
%                 net_init.wIn(net.oUind,:) = [0 .33];
                tau_trans = 1; % now controls wJ learning rate
            end
            
            filt_scale = 50; % plant scale factor currently

            % stim scalar determines whether a control (0) or lick- (-1) or lick+ (1) perturbation experiments
            stim = stim_list(g);
            [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale);

            run(g).output = output;
            run(g).net = net_out;
            run(g).pred_da_sense = pred_da_sense;
            run(g).pred_da_move = pred_da_move;
            disp(['Completed run: ' num2str(g)]);
        end
        
    case 'variable_cue'
        [output,net] = dlRNN_train_learnDA(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,1250);

    otherwise
        
end

% return: net (trained network) and output (end of training simulation)

%% Summary display plot for talks of training experience.
clear model model_runs;
[err_map] = TNC_CreateRBColormap(numel(run),'cpb'); 
[stim_map] = [1 0 0.67 ; 0 1 0.67 ; 0 0.67 1]
cost = 500;
% visualize training error:
latency_cost = cost * (1-exp(-[0:1:1500]/250)');
figure(3); hold on; plot(latency_cost);

% anticip_cost =  cost * (([0:800]-200)./800).^2 + ... % component that is effort-like cost
%     cost * 0.21 * exp(-([0:800]-200)./120);    
anticip_cost =  0.33*polyval(emp_ant_cost,0:0.1:9) + ... % component that is effort-like cost
    cost * 0.36 * exp(-([0:0.1:9]-2)./1.2);    

tmptmp = (latency_cost*ones(1,numel(anticip_cost)) + (ones(numel(latency_cost),1)*anticip_cost));

[resp_map] = TNC_CreateRBColormap(8,'cpb');
    figure(501); clf;
%             imagesc(anticip_cost,latency_cost,tmptmp); colormap(err_map); hold on;
            imagesc(0:0.1:9,0:1:1500,tmptmp); 
            colormap(err_map); hold on;

set(gca,'YDir','normal');
            hold on;
            hold on;

            
for g=1:numel(run)
    
    final_output_layer(g,:) = run(g).net.J(1,:);
    final_lat(g,1) = run(g).output.pass(400).lat;
    
end
[vvv,iii] = sort(final_lat);
figure(510); subplot(1,5,1:4); imagesc(final_output_layer(iii,:));
figure(510); subplot(1,5,5); imagesc(final_lat(iii,1));

    anticip = [];
    latency = [];
    sens_gain = [];

    anticip_u = [];
    latency_u = [];
    sens_gain_u = [];
    
for g=1:numel(run)
    
    trials_to_criterion(g) = numel(run(g).output.pass);
        
    model(g).anticip = [];
    model(g).latency = [];
    model(g).sens_gain = [];

    model(g).anticip_u = [];
    model(g).latency_u = [];
    model(g).sens_gain_u = [];
    
    for kk=1:numel(run(g).output.pass)
        if mod(kk,run(g).net.update) == 0 || kk == 1
            model(g).anticip = [model(g).anticip run(g).output.pass(kk).anticip];
            model(g).latency = [model(g).latency run(g).output.pass(kk).lat];
            model(g).sens_gain = [model(g).sens_gain run(g).output.pass(kk).sens_gain];
            
            anticip = [anticip run(g).output.pass(kk).anticip];
            latency = [latency run(g).output.pass(kk).lat];
            sens_gain = [sens_gain run(g).output.pass(kk).sens_gain];

            model(g).anticip_u = [model(g).anticip_u run(g).output.pass(kk).anticip_u];
            model(g).latency_u = [model(g).latency_u run(g).output.pass(kk).lat_u];
            model(g).sens_gain_u = [model(g).sens_gain_u run(g).output.pass(kk).sens_gain_u];
            
            anticip_u = [anticip_u run(g).output.pass(kk).anticip_u];
            latency_u = [latency_u run(g).output.pass(kk).lat_u];
            sens_gain_u = [sens_gain_u run(g).output.pass(kk).sens_gain_u];
            
        end
    end
    
    model(g).latency(model(g).latency>1500) = 1500;
    latency(latency>1500) = 1500;
    
    figure(501);
%     plot(sgolayfilt(model(g).anticip,3,21),sgolayfilt(model(g).latency,3,21),'color',err_map(g,:)./2); hold on;
    model_runs.anticip(g,:) = model(g).anticip;
    model_runs.latency(g,:) = model(g).latency;
    
%     plot(sgolayfilt(model(g).anticip,3,21),sgolayfilt(model(g).latency,3,21),'color',stim_map(stim_list(g)+2,:)); hold on;
%     figure(502); subplot(4,6,g); imagesc(run(g).pred_da_move+100*run(g).pred_da_sense);

end

lick_counts = unique(anticip);
for pp=1:numel(lick_counts)
    model_runs.anticip_d(pp) = lick_counts(pp);
    model_runs.latency_d(pp) = mean(latency(find(anticip==lick_counts(pp))));
end

lick_counts_u = unique(anticip_u);
for pp=1:numel(lick_counts_u)
    model_runs.anticip_d_u(pp) = lick_counts_u(pp);
    model_runs.latency_d_u(pp) = mean(latency_u(find(anticip_u==lick_counts_u(pp))));
end

figure(701);
stim_cat = [-1 0 1];
    for sg = 1:3
        inds = find(stim_list==stim_cat(sg));
%         plot(sgolayfilt(mean(model_runs.anticip(inds,:)),3,21),sgolayfilt(mean(model_runs.latency(inds,:)),3,21),'color',stim_map(sg,:),'linewidth',3); hold on;
        plot(sgolayfilt(mean(model_runs.anticip(inds,:)),3,7),sgolayfilt(mean(model_runs.latency(inds,:)),3,7),'color',[1 1 1],'linewidth',3); hold on;
    end
%     plot(model_runs.anticip_d , model_runs.latency_d ,'w', 'linewidth', 3 ); hold on;
            title('Cost surface'); ylabel('Latency (ms)'); xlabel('Anticipatory licks'); box off;

figure(601); clf;
            
    for sg = 1:3
        inds = find(stim_list==stim_cat(sg));
     plot(0:5:800,sgolayfilt(mean(model_runs.anticip(inds,:)),3,7),'color',stim_map(sg,:),'linewidth',2); hold on;
    end
    legend('Lick-','Control','Lick+');
ylabel('Anticipatory licks'); xlabel('Training trials'); 
all_latency = zeros(numel(run) , 200); 
all_latency_u = zeros(numel(run) , 200); 

for kk=1:numel(run)
    all_latency(kk,1:numel(model(kk).latency)) = model(kk).latency(1,1:numel(model(kk).latency));
    all_latency_u(kk,1:numel(model(kk).latency_u)) = model(kk).latency_u(1,1:numel(model(kk).latency_u));
end
box off;

figure(503); clf;
    shadedErrorBar( [1:200]*run(1).net.update  , mean(all_latency,1) , std(all_latency,[],1)./sqrt(size(all_latency,1)) , {'color',[1 0.1 0.2]}); hold on;
    shadedErrorBar( [1:200]*run(1).net.update , mean(all_latency_u,1) , std(all_latency_u,[],1)./sqrt(size(all_latency_u,1)) ); hold on;
            ylabel('Latency to collect reward (ms)'); xlabel('Training trial bins');
%             legend('Cued','Uncued');
axis([0 200 0 1500]);

figure(500); boxplot(trials_to_criterion); ylabel('Trials to criterion');

%% Examine stim Lick-/+ predictions

figure(601); clf;
stim_cat = [-1 0 1];
            
for sg = 1:3
    inds = find(stim_list==stim_cat(sg));
    
    switch sg
        
        case 1
            stim_minus = model_runs.anticip(inds,:);
            
        case 2
            stim_cntrl = model_runs.anticip(inds,:);
            
        case 3
            stim_plus = model_runs.anticip(inds,:);
            
    end
    
end


[stim_minus_b] = TNC_BinAndMean([1 5:5:800],mean(stim_minus)-mean(stim_cntrl),8);
[stim_minus_berr] = TNC_BinAndMean([1 5:5:800],std(stim_minus)./sqrt(9),8);
% plot(stim_minus_b.bins.center,stim_minus_b.bins.avg,'color',[57 181 74]/255,'linewidth',2); hold on;
shadedErrorBar(stim_minus_b.bins.center,stim_minus_b.bins.avg,stim_minus_berr.bins.avg,{'color',[57 181 74]/255}); hold on;

[stim_plus_b] = TNC_BinAndMean([1 5:5:800],mean(stim_plus)-mean(stim_cntrl),8);
[stim_plus_berr] = TNC_BinAndMean([1 5:5:800],std(stim_plus)./sqrt(9),8);
% plot(stim_plus_b.bins.center,stim_plus_b.bins.avg,'color',[146 39 143]/255,'linewidth',2); hold on;
shadedErrorBar(stim_plus_b.bins.center,stim_plus_b.bins.avg,stim_plus_berr.bins.avg,{'color',[146 39 143]/255}); hold on;

% shadedErrorBar(0:5:800,,std(stim_minus)./sqrt(9),{'color',[57 181 74]/255}); hold on;
% shadedErrorBar(0:5:800,mean(stim_plus)-mean(stim_cntrl),std(stim_plus)./sqrt(9),{'color',[146 39 143]/255}); hold on;
plot([0 800],[0 0],'k-');

% legend('Lick-','Lick+');
ylabel('Cued licks - Cntrl (Hz)'); xlabel('Training trials'); 
axis([0 800 -2.5 2.5]);
all_latency = zeros(numel(run) , 200); 
all_latency_u = zeros(numel(run) , 200); 
box off;

%% Fitiing procedure to find params for experimental learning curves
stim_list = zeros(1,12);
scales = repmat([30 40 ],1,2);
tau_vec = reshape(ones(2,1)*[20 30],1,4);
% scales = 30;
% tau_vec = 30;
% scales = [1.05 1.06 1.07 1.09]
% tau_vec = ones(1,4)*50

clear all_lat* grid_run

global monitor;
monitor = 0;
disp('_ New run');

% indices=[ii(212) ii(213) ii(205) ii(196) ii(176) ii(70)];

index=ii(89);

% optimal is rough;y plant_scale=60 tau=100 for pass-thru plant model
%optimal is filt_scale=1.07 tau=50 for high-pass plant model

% for grid_i = 1:numel(tau_vec)
for grid_i = 1:numel(scales)
        
        disp('_ ');
        disp(['Training grid ' num2str(grid_i) ' out of ' num2str(numel(scales)) '... ']);
        filt_scale = scales(grid_i);
        tau_trans = tau_vec(grid_i);
        clear run;

        parfor g = 1:numel(stim_list)

%             index = indices(g);
            net_init = gens.dets(index).net % diverse initial states
            net_init.wIn(net.oUind,:) = [0 0];
            stim = stim_list(g);

            [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale);

            run(g).output = output;
            run(g).net = net_out;
            run(g).pred_da_sense = pred_da_sense;
            run(g).pred_da_move = pred_da_move;
            disp(['Completed run: ' num2str(g) ' from grid ' num2str(grid_i)]);
            
        end

        for gg=1:numel(run)
                
            latency = [];
            latency_u = [];

            for kk=1:numel(run(gg).output.pass)
                if mod(kk,run(gg).net.update) == 0 || kk == 1
                    latency = [latency run(gg).output.pass(kk).lat];
                    latency_u = [latency_u run(gg).output.pass(kk).lat_u];                        
                end
            end

            all_latency(gg,:) = latency;
            all_latency_u(gg,:) = latency_u;
            
        end
            

        grid_run(grid_i).all_latency = all_latency;
        grid_run(grid_i).all_latency_u = all_latency_u;
        grid_run(grid_i).filt_scale = filt_scale;
        grid_run(grid_i).tau_trans = tau_trans;
        
end

figure(100); clf; clear to_fit;

% To compare to data I need to take the mean of the first point and then
% average over groups of 20 points (100 trials)
for qq=1:numel(scales)

    to_fit.cued(qq,1) = mean(grid_run(qq).all_latency(:,1));
    to_fit.un(qq,1) = mean(grid_run(qq).all_latency_u(:,1));
    
    for q=1:8 % 800 trials total / 100 trial epochs
        
        inds = ((q-1)*20)+2:q*20+1;

        to_fit.cued(qq,q+1) = mean( mean( grid_run(qq).all_latency(:,inds) ) );
        to_fit.un(qq,q+1) = mean( mean( grid_run(qq).all_latency_u(:,inds) ) );

%         to_fit.cued_e(qq,q+1) = std( mean( grid_run(qq).all_latency(:,inds) ) ) ./ sqrt(numel(stim_list));
%         to_fit.un_e(qq,q+1) = std( mean( grid_run(qq).all_latency_u(:,inds) ) ) ./ sqrt(numel(stim_list));
        to_fit.cued_e(qq,q+1) = std( mean( grid_run(qq).all_latency(:,inds) ) );
        to_fit.un_e(qq,q+1) = std( mean( grid_run(qq).all_latency_u(:,inds) ) );
        
    end
    
    subplot(sqrt(numel(scales)),sqrt(numel(scales)),qq);
%     plot(0:100:800,to_fit.cued(qq,:),'r'); hold on; plot(0:100:800,to_fit.un(qq,:),'k');
    shadedErrorBar(0:100:800,to_fit.cued(qq,:),to_fit.cued_e(qq,:),{'color',[1 0 0]}); hold on; 
    shadedErrorBar(0:100:800,to_fit.un(qq,:),to_fit.un_e(qq,:));
    box off;
%     plot([100:100:800]-0,mean(lats.cued),'r-o','MarkerFace','r'); hold on; plot([100:100:800]-0,mean(lats.uncued,'omitnan'),'k-o','MarkerFace','k');
    title(['Scales: ' num2str(grid_run(qq).filt_scale) ';  Tau: ' num2str(grid_run(qq).tau_trans)]);
end

% Run used in figure saved as: Coddington-ModelFigure-CuedUncuedInitData

%% Examine how the balance of transient learning (eta_wIn) and sustained learning (eta_J) multipliers influence the course of learning

% If stim != [-1,0,1] then it is just a multiplier applied on every trial
stim_list = repmat([0.2 0.5 0 2 5],1,6);

% Converting tau_trans into a multiplier for eta_J that can be passed into dlRNN_train_learn
tau_vec = [0.2 0.5 1 2 5]

% Fix the plant scales, not really relevant to this examination
scales = 50*ones(1,numel(tau_vec));

clear all_* grid_run

global monitor;
monitor = 0;
disp('_ New run');

% index=ii(89);
index=ii(101);
% index = ii(213);

% for grid_i = 1:numel(tau_vec)
for grid_i = 1:numel(scales)
        
        disp('_ ');
        disp(['Training grid ' num2str(grid_i) ' out of ' num2str(numel(scales)) '... ']);
        filt_scale = scales(grid_i);
        tau_trans = tau_vec(grid_i);
        clear run;

        parfor g = 1:numel(stim_list)

            stim = stim_list(g);
            tau_trans = 1;

% Un -comment this if one wants to examine a replica of several runs. Used
% for DA correlate predictions in Figure 4.

% comment
for ggg=1:20
    monitor =0;
    stim = 0;
    filt_scale = 50;
% comment
            net_init = gens.dets(index).net;
            net_init.wIn(net.oUind,:) = [0 0];
            

            [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u,pred_da_move_o,pred_da_sense_o] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale);
            
% comment
    da_store(ggg).pred_da_move = pred_da_move;
    da_store(ggg).pred_da_move_u = pred_da_move_u;
    da_store(ggg).pred_da_sense = pred_da_sense;
    da_store(ggg).pred_da_sense_u = pred_da_sense_u;
    da_store(ggg).pred_da_move_o = pred_da_move_o;
    da_store(ggg).pred_da_sense_o = pred_da_sense_o;
    
    run(ggg).output = output;
    run(ggg).net = net_out;
    run(ggg).stim = stim;

    ggg
end
% comment

            run(g).output = output;
            run(g).net = net_out;
            run(g).stim = stim;
            run(g).pred_da_sense = pred_da_sense;
            run(g).pred_da_move = pred_da_move;
            disp(['Completed run: ' num2str(g) ' from grid ' num2str(grid_i)]);
            
        end

        for gg=1:numel(run)
                
            latency = []; trans_r=[]; trans_c=[]; sust=[];            

            for kk=1:numel(run(gg).output.pass)
                if mod(kk,run(gg).net.update) == 0 || kk == 1
                    latency = [latency run(gg).output.pass(kk).lat];
                    trans_r = [trans_r run(gg).output.pass(kk).trans_r ];
                    trans_c = [trans_c run(gg).output.pass(kk).trans_c ];
                    sust    = [sust run(gg).output.pass(kk).sust ];
                end
            end

            all_latency(gg,:) = latency;
            all_trans_r(gg,:) = trans_r;
            all_trans_c(gg,:) = trans_c;
            all_sust(gg,:) = sust;
            
        end
            

        grid_run(grid_i).all_latency = all_latency;
        grid_run(grid_i).trans_r = all_trans_r;
        grid_run(grid_i).trans_c = all_trans_c;
        grid_run(grid_i).sust = all_sust;

        grid_run(grid_i).filt_scale = filt_scale;
        grid_run(grid_i).tau_trans = tau_trans;
        
        grid_run
        
end

save ~/'Dropbox (HHMI)'/'Conditioned responding and DA'/Figures/Fig4_Model/Coddington-ModelFigure-LearningRateCompare-488-wIn-30-eJ-2p5e5 grid_run;

%% check a single run

latency = []; trans_r=[]; trans_c=[]; sust=[];            

            for kk=1:numel(output.pass)
                if mod(kk,net_out.update) == 0 || kk == 1
                    latency = [latency output.pass(kk).lat ];
                    trans_r = [trans_r output.pass(kk).trans_r ];
                    trans_c = [trans_c output.pass(kk).trans_c ];
                    sust    = [sust    output.pass(kk).sust ];
                end
            end
            
            figure(5); hold on; plot3(sgolayfilt(sust,3,11),-sgolayfilt(trans_c,3,11),1-exp(-sgolayfilt(latency,3,11)/500)); grid on; box on;
            
%% Use a single simulation run to estimate DA transients
% Specific to the way that some iterations of simulations were run

clear da_pred;

% as per fig 2
early = 2:20;

% % learning extent match
% mid = 60:80;
late = 80:100;

% exact match
mid = 40:60;
% late = 120:160;

s_scl = 2;
m_scl = 1;

num_sims = numel(da_store);

jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
for ggg=1:num_sims
    da_pred.early.c(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(early,:)) + m_scl*mean(da_store(ggg).pred_da_move(early,:)) , kern , 'same');
    da_pred.early.u(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense_u(early,:)) + m_scl*mean(da_store(ggg).pred_da_move_u(early,:)) , kern , 'same');

    da_pred.mid.c(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(mid,:)) + m_scl*mean(da_store(ggg).pred_da_move(mid,:)) , kern , 'same');
    da_pred.mid.u(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense_u(mid,:)) + m_scl*mean(da_store(ggg).pred_da_move_u(mid,:)) , kern , 'same');
    da_pred.mid.o(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense_o(mid,:)) + m_scl*mean(da_store(ggg).pred_da_move_o(mid,:)) , kern , 'same');

    da_pred.late.c(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(late,:)) + m_scl*mean(da_store(ggg).pred_da_move(late,:)) , kern , 'same');
    da_pred.late.u(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense_u(late,:)) + m_scl*mean(da_store(ggg).pred_da_move_u(late,:)) , kern , 'same');
    da_pred.late.o(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense_o(late,:)) + m_scl*mean(da_store(ggg).pred_da_move_o(late,:)) , kern , 'same');
end

% figure(500); clf;
% subplot(1,3,1); plot(da_pred.early.c,'r','linewidth',2); hold on; plot(da_pred.early.u,'k','linewidth',2); axis([0 3e3 -0.001 0.01]); box off;
% subplot(1,3,2); plot(da_pred.mid.c,'r','linewidth',2); hold on; plot(da_pred.mid.u,'k','linewidth',2); axis([0 3e3 -0.001 0.01]);  box off;
% subplot(1,3,3); plot(da_pred.late.c,'r','linewidth',2); hold on; plot(da_pred.late.u,'k','linewidth',2); plot(da_pred.late.o,'b','linewidth',2); axis([0 3e3 -0.001 0.01]);  box off;

figure(500); clf;
subplot(1,3,1); shadedErrorBar(-1600:1399,mean(da_pred.early.u),std(da_pred.early.u)./sqrt(num_sims),{'color','k'}); hold on; shadedErrorBar(-1600:1399,mean(da_pred.early.c),std(da_pred.early.c)./sqrt(num_sims),{'color','r'}); axis([-1600 1.4e3 -0.001 0.0075]); box off;
subplot(1,3,2); shadedErrorBar(-1600:1399,mean(da_pred.mid.u),std(da_pred.mid.u)./sqrt(num_sims),{'color','k'}); hold on; shadedErrorBar(-1600:1399,mean(da_pred.mid.o),std(da_pred.mid.o)./sqrt(num_sims),{'color','b'}); shadedErrorBar(-1600:1399,mean(da_pred.mid.c),std(da_pred.mid.c)./sqrt(num_sims),{'color','r'}); axis([-1600 1.4e3 -0.001 0.0075]);  box off;
subplot(1,3,3); shadedErrorBar(-1600:1399,mean(da_pred.late.u),std(da_pred.late.u)./sqrt(num_sims),{'color','k'});  hold on; shadedErrorBar(-1600:1399,mean(da_pred.late.o),std(da_pred.late.o)./sqrt(num_sims),{'color','b'}); shadedErrorBar(-1600:1399,mean(da_pred.late.c),std(da_pred.late.c)./sqrt(num_sims),{'color','r'});axis([-1600 1.4e3 -0.001 0.0075]);  box off;
xlabel('Time from reward (ms)');

save ~/'Dropbox (HHMI)'/'Conditioned responding and DA'/Figures/Fig4_Model/Figure-DataForDAPrediction da_pred da_store



% Estimate DA transients for lick+ vs lick-

jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);

anticip_licks_plot = zeros(num_sims,numel(mon_inds));

for ggg=1:num_sims
    run_ind = ggg;

    mon_inds = [1 5:5:800 ];
    anticip_licks = zeros(1,numel(mon_inds));
    cnt = 1;
    for gg=mon_inds
        anticip_licks(cnt) = numel(find(run(run_ind).output.pass(gg).chk.v > 900 & run(run_ind).output.pass(gg).chk.v < 1600));
%         anticip_licks(cnt) = run(run_ind).output.pass(gg).anticip;
        anticip_licks_plot(ggg,cnt) = anticip_licks(cnt)>0;
        cnt = cnt+1;
    end
    
    anticip_licks_plot(ggg,:) = sgolayfilt(anticip_licks_plot(ggg,:),3,11);

    trial_100_200_lickM = (find(anticip_licks(1:61)<1));
    trial_100_200_lickP = (find(anticip_licks(1:61)>1));
    trial_400_800_lickM = (find(anticip_licks(81:161)<1))+80;
    trial_400_800_lickP = (find(anticip_licks(81:161)>1))+80

    da_pred.init.lm(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(trial_100_200_lickM,:)) + m_scl*mean(da_store(ggg).pred_da_move(trial_100_200_lickM,:)) , kern , 'same');
    da_pred.init.lp(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(trial_100_200_lickP,:)) + m_scl*mean(da_store(ggg).pred_da_move(trial_100_200_lickP,:)) , kern , 'same');
    da_pred.end.lm(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(trial_400_800_lickM,:)) + m_scl*mean(da_store(ggg).pred_da_move(trial_400_800_lickM,:)) , kern , 'same');
    da_pred.end.lp(ggg,:) = conv( s_scl*mean(da_store(ggg).pred_da_sense(trial_400_800_lickP,:)) + m_scl*mean(da_store(ggg).pred_da_move(trial_400_800_lickP,:)) , kern , 'same');
end

figure(501); clf;
subplot(1,2,1); shadedErrorBar(-1600:1399,mean(da_pred.init.lm),std(da_pred.init.lm)./sqrt(num_sims),{'color','r'}); hold on; shadedErrorBar(-1600:1399,mean(da_pred.init.lp),std(da_pred.init.lp)./sqrt(num_sims),{'color',[0 0.33 0.5]}); axis([-1600 1.4e3 -0.001 0.0075]); box off;
title('Trials 1-300'); axis([-1600 1.4e3 -0.001 0.0075]); box off;
subplot(1,2,2); shadedErrorBar(-1600:1399,mean(da_pred.end.lm,'omitnan'),std(da_pred.end.lm,'omitnan')./sqrt(num_sims),{'color','r'}); hold on; shadedErrorBar(-1600:1399,mean(da_pred.end.lp),std(da_pred.end.lp)./sqrt(num_sims),{'color',[0 0.33 0.5]}); axis([-1600 1.4e3 -0.001 0.0075]); box off;
title('Trials 400-800'); axis([-1600 1.4e3 -0.001 0.0075]); box off;
xlabel('Time from reward (ms)');

figure(502);
shadedErrorBar(mon_inds,mean(anticip_licks_plot),std(anticip_licks_plot)./sqrt(num_sims),{'color',[0.25 0.25 0.25]});
axis([0 800 0 1]); box off;

%% Plot the scan over learning rates simulations
figure(101); clf; clear to_fit; figure(102); clf; figure(103); clf;
clear lrn_traj

order = 3;
width = 11;
win_ln = 24;
bin_log='fit';
trial_krn = [0 ones(1,win_ln)/win_ln 0];

% % DIMENSIONS
% 1: tau_vec conditions
% 2: trials
% 3: stim_list conditions

% To compare to data I need to take the mean of the first point and then
% average over groups of 20 points (100 trials)
for qq=1:numel(scales)
    
    for q=1:5 % stim_list items
        
        inds = [0:5:25]+q; % demix the repeating structure of stim_vec

        switch bin_log
            case 'bin'
                [tb] = TNC_BinAndMean(1:160,mean( grid_run(qq).all_latency(inds,2:end) ),win_ln);        
                lrn_traj.lat(qq,:,q) = tb.bins.avg;
                lrn_traj.cost(qq,:,q) = 1-exp(-tb.bins.avg/500);

                [tb] = TNC_BinAndMean(1:160,mean( grid_run(qq).trans_r(inds,2:end) ),win_ln);        
                lrn_traj.trans_r(qq,:,q) = tb.bins.avg; 


                [tb] = TNC_BinAndMean(1:160,mean( grid_run(qq).trans_c(inds,2:end) ),win_ln);        
                lrn_traj.trans_c(qq,:,q) = tb.bins.avg;

                [tb] = TNC_BinAndMean(1:160,mean( grid_run(qq).sust(inds,2:end) ),win_ln);        
                lrn_traj.sust(qq,:,q) = tb.bins.avg;
                
            case 'smooth'            
                lrn_traj.lat(qq,:,q) = sgolayfilt( mean( grid_run(qq).all_latency(inds,2:end) ) , order, width);
                lrn_traj.cost(qq,:,q) = 1-exp(-lrn_traj.lat(qq,:,q)/500);

                lrn_traj.trans_r(qq,:,q) = sgolayfilt( mean( grid_run(qq).trans_r(inds,2:end) ) , order, width);            
                lrn_traj.trans_c(qq,:,q) = sgolayfilt( mean( grid_run(qq).trans_c(inds,2:end) ) , order, width);
                lrn_traj.sust(qq,:,q) = sgolayfilt( mean( grid_run(qq).sust(inds,2:end) ) , order, width);    
                
            case 'fit'
                x = 1:numel(data);
                data = mean( grid_run(qq).all_latency(inds,2:end) );
                latency_model = fit(x',data'-100,'exp1','Lower',[1 -0.1],'Upper',[Inf 0]);
                lrn_traj.lat(qq,:,q) = latency_model(x);
                lrn_traj.cost(qq,:,q) = 1-exp(-lrn_traj.lat(qq,:,q)/500);
                
                data = mean( grid_run(qq).trans_r(inds,2:end) );
                trans_r_model = fit(x',(data'-10)/10,'exp1','Lower',[1 -0.1],'Upper',[Inf 0]);
                lrn_traj.trans_r(qq,:,q)  = (data'-10)/10 ; %trans_r_model(x);

                data = mean( grid_run(qq).sust(inds,2:end) );
                sust_model = fit(x',data'-1,'exp1','Lower',[-Inf -0.1],'Upper',[-1 0]);
                lrn_traj.sust(qq,:,q) = sust_model(x);          

                lrn_traj.trans_c(qq,:,q) = sgolayfilt( mean( grid_run(qq).trans_c(inds,2:end) ) , order, width);

        end

        if qq==5 & q==5 | qq==5 & q==1 | q==5 & qq==1 | q==1 & qq==1
            figure(101); plot3( lrn_traj.sust(qq,:,q) , -lrn_traj.trans_r(qq,:,q)/10 , lrn_traj.cost(qq,:,q) , 'linewidth' , 4 , 'color' , [0.2*q , 0.33 , 0.2*qq 1]); hold on; box on;grid on;
        elseif qq==3 & q==3
            figure(101); plot3( lrn_traj.sust(qq,:,q) , -lrn_traj.trans_r(qq,:,q)/10 , lrn_traj.cost(qq,:,q) , 'linewidth' , 4 , 'color' , [0 0 0]); hold on; box on;grid on;
        else
            figure(101); plot3( lrn_traj.sust(qq,:,q) , -lrn_traj.trans_r(qq,:,q)/10 , lrn_traj.cost(qq,:,q) , 'linewidth' , 1 , 'color' , [0.2*q , 0.33 , 0.2*qq 0.75]); hold on; box on;grid on;
        end
        xlabel('Sustained'); ylabel('Transient'); zlabel('Cost'); view(48,30);
            
        if qq==3 & q==3
            figure(102); loglog(tau_vec(q),tau_vec(qq),'.','MarkerSize',90,'color' , [0 0 0]); hold on; xlabel('Trans. rate factor'); ylabel('Sust. rate factor'); axis([0.1 10 0.1 10]); box off;            
        else
            figure(102); loglog(tau_vec(q),tau_vec(qq),'.','MarkerSize',90,'color' , [0.2*q , 0.33 , 0.2*qq 1]); hold on; xlabel('Trans. rate factor'); ylabel('Sust. rate factor'); axis([0.1 10 0.1 10]); box off;
        end
        
        figure(103); 
        if qq==3 & q==3
            subplot(311); plot( lrn_traj.sust(qq,:,q) , 'linewidth' , 4 , 'color' , [0 0 0]); hold on; box off; ylabel('Sustained');
            subplot(312); plot( -lrn_traj.trans_r(qq,:,q) , 'linewidth' , 4 , 'color' , [0 0 0]); hold on; box off; ylabel('-Transient');
            subplot(313); plot( lrn_traj.lat(qq,:,q) , 'linewidth' , 4 , 'color' , [0 0 0]); hold on; box off; ylabel('Latency');
        else
            subplot(311); plot( lrn_traj.sust(qq,:,q) , 'linewidth' , 2 , 'color' , [0.2*q , 0.33 , 0.2*qq 0.75]); hold on; box off; ylabel('Sustained');
            subplot(312); plot( -lrn_traj.trans_r(qq,:,q) , 'linewidth' , 2 , 'color' , [0.2*q , 0.33 , 0.2*qq 0.75]); hold on; box off; ylabel('-Transient');
            subplot(313); plot( lrn_traj.lat(qq,:,q) , 'linewidth' , 2 , 'color' , [0.2*q , 0.33 , 0.2*qq 0.75]); hold on; box off; ylabel('Latency');
        end
    end
    
end

%% Calculate predicted DA transients

model_pred_da_move = zeros(2000./net_out.update,size(input{1}(:,:),2));
model_pred_da_sense = zeros(2000./net_out.update,size(input{1}(:,:),2));

for g=1:numel(run)

    model_pred_da_move(1:size(run(g).pred_da_move,1),:) =  model_pred_da_move(1:size(run(g).pred_da_move,1),:) + run(g).pred_da_move;
    model_pred_da_sense(1:size(run(g).pred_da_sense,1),:) = model_pred_da_sense(1:size(run(g).pred_da_sense,1),:) + run(g).pred_da_sense;

    model_pred_da_move_u(1:size(run(g).pred_da_move,1),:) =  model_pred_da_move_u(1:size(run(g).pred_da_move_u,1),:) + run(g).pred_da_move_u;
    model_pred_da_sense_u(1:size(run(g).pred_da_sense,1),:) = model_pred_da_sense_u(1:size(run(g).pred_da_sense_u,1),:) + run(g).pred_da_sense_u;
    
end

figure(700); subplot(121); imagesc(model_pred_da_move);
figure(700); subplot(122); imagesc(model_pred_da_sense);

da_resp.early.move = sum(model_pred_da_move(1:50,:),1)./50;
da_resp.early.sens = sum(model_pred_da_sense(1:50,:),1)./50;

da_resp.mid.move = sum(model_pred_da_move(50:150,:),1)./100;
da_resp.mid.sens = sum(model_pred_da_sense(50:150,:),1)./100;

da_resp.late.move = sum(model_pred_da_move(300:400,:),1)./100;
da_resp.late.sens = sum(model_pred_da_sense(300:400,:),1)./100;

figure(701); clf;
subplot(131);  plot([-1599:1400],da_resp.early.sens+da_resp.early.move,'k-','linewidth',2); hold on;
    plot([-1599:1400],da_resp.early.move); hold on; plot([-1599:1400],da_resp.early.sens);axis tight; box off; axis([-1600 1400 -0.05 0.1]);
    ylabel('Predicted DA response (au)'); xlabel('Time from reward (ms)'); title('Early (1-200)');
subplot(132); plot([-1599:1400],da_resp.mid.sens+da_resp.mid.move,'k-','linewidth',2); hold on;
    plot([-1599:1400],da_resp.mid.move); hold on; plot([-1599:1400],da_resp.mid.sens); axis tight; box off;axis([-1600 1400 -0.05 0.1]);
    ylabel('Predicted DA response (au)'); xlabel('Time from reward (ms)');  title('Middle (400-800)');
subplot(133); plot([-1599:1400],da_resp.late.sens+da_resp.late.move,'k-','linewidth',2); hold on;
    plot([-1599:1400],da_resp.late.move); hold on; plot([-1599:1400],da_resp.late.sens); axis tight; box off;axis([-1600 1400 -0.05 0.1]); 
    ylabel('Predicted DA response (au)'); xlabel('Time from reward (ms)'); title('Late (1600-2000)');

%%
for g=1:numel(run)
    
    model.anticip = [];
    model.latency = [];
    model.sens_gain = [];

    for kk=1:numel(output.pass)
        if mod(kk,net.update) == 0 || kk == 1
            model.anticip = [model.anticip run(g).output.pass(kk).anticip];
            model.latency = [model.latency run(g).output.pass(kk).lat];
            model.sens_gain = [model.sens_gain run(g).output.pass(kk).sens_gain];
        end
    end

    pred_da_sumRewResp = sum(run(g).pred_da_sense(:,1600:2000),2) + 0.01*sum(run(g).pred_da_move(:,1200:2000),2);
    pred_da_sumCueResp = sum(run(g).pred_da_sense(:,100:500),2) + 0.01*sum(run(g).pred_da_move(:,100:500),2);

    figure(300); clf;
    plot(net.update*[1:size(pred_da_sense,1)],pred_da_sumRewResp); 
    hold on;
    plot(net.update*[1:size(pred_da_sense,1)],pred_da_sumCueResp); 
     box off;
     legend('Reward resp.','Cue resp.');
     xlabel('Trials'); ylabel('Predict DA activity (a.u.)');
    % 
    % two subplots. 
    % on left is model output: 
    % latency to collect vs. anticipatory licking
    % colorized by trials
    figure(200+(g*10)); clf;
    subplot(121);
    scatter3(learn.smooth.anticip,learn.smooth.sens_gain,learn.smooth.latency,40,learn.smooth.dopaR,'filled'); colormap(cmap_learn);
        axis([-50 750 0 30 -10 180]);
        xlabel('Anticipatory licking'); ylabel('Sensory lag'); zlabel('Collect latency');
    title('DATA');

        subplot(122); hold off;
    % plot3(model.anticip,-model.sens_gain,model.latency/10,'k-','color',[0.8 0.8 0.8]); hold on;
    scatter3(model.anticip,-model.sens_gain,model.latency/10,40,pred_da_sumRewResp','filled'); colormap(resp_map);
    grid on;
        axis([0 10 -10e-3 0 -10 180]);
        xlabel('Anticipatory licking'); ylabel('Sensory lag'); zlabel('Collect latency');
    title('MODEL');

    % on right is same but for example behavior data from a single mouse

    figure(201+(g*10))
    subplot(121);
    cmap_sess = cmap_sess(1:9,:);
    scatter3(learn.smooth.anticip,learn.smooth.sens_gain,learn.smooth.latency,40,sess_inds,'filled'); colormap(cmap_learn);
        axis([-50 750 0 30 -10 180]);
        xlabel('Anticipatory licking'); ylabel('Sensory lag'); zlabel('Collect latency');
    title('DATA');


    subplot(122); hold off;
    % plot3(model.anticip,-model.sens_gain,model.latency/10,'k-','color',[0.8 0.8 0.8]); hold on;
    scatter3(model.anticip,-model.sens_gain,model.latency/10,40,1:numel(model.anticip),'filled'); colormap(resp_map);
    grid on;
        axis([0 10 -10e-3 0 -10 180]);
        xlabel('Anticipatory licking'); ylabel('Sensory gain'); zlabel('Collect latency');
    title('MODEL');
end
% colorized by some version of DA activity

%% Examine good learner vs bad learner simulations

clear model;

% Bad learner is defined as higher init transient component, but lower eta_wIn (can imagine a homeostatic mechanism might produce this effect)
% need to add predicted DA signal

% as per fig 3
early = 2:20;
late = 141:161;

s_scl = 2;
m_scl = 0.5;


jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);

for g=1:numel(run)
    
    trials_to_criterion(g) = numel(run(g).output.pass);
        
    model(g).anticip = [];
    model(g).latency = [];
    model(g).sens_gain = [];

    model(g).anticip_u = [];
    model(g).latency_u = [];
    model(g).sens_gain_u = [];
    
    for kk=1:numel(run(g).output.pass)
        if mod(kk,run(g).net.update) == 0 || kk == 1
            
            tmp_lat = run(g).output.pass(kk).lat;
            if tmp_lat > 1000
                tmp_lat = 1000;
            end
            model(g).anticip = [model(g).anticip run(g).output.pass(kk).anticip];
            model(g).latency = [model(g).latency tmp_lat];
            
            model(g).da_100 = conv( s_scl*mean(run(g).pred_da_sense(early,:)) + m_scl*mean(run(g).pred_da_move(early,:)) , kern , 'same');
            model(g).da_600 = conv( s_scl*mean(run(g).pred_da_sense(late,:)) + m_scl*mean(run(g).pred_da_move(late,:)) , kern , 'same');
        end
    end
    
    latency(g,:) = model(g).latency;
    anticip(g,:) = model(g).anticip;
    da_early(g,:) = model(g).da_100;
    da_late(g,:) = model(g).da_600;            

end

% calculate DA differences and latency differences per init

num_init = numel(unique(inits));

for g=1:num_init
    
    good_i = g:num_init:numel(stim_list)/2;
    bad_i = numel(stim_list)/2+g:num_init:numel(stim_list);
    
    corr_fig3.da(g) = max(mean(da_early(good_i,1500:end))) ;
    corr_fig3.lat(g) = mean(mean(latency(good_i,end-40:end)) );
    corr_fig3.da(g+num_init) = max(mean(da_early(bad_i,1500:end))) ;
    corr_fig3.lat(g+num_init) = mean(mean(latency(bad_i,end-40:end))) ;
    
end

individs = TNC_CreateRBColormap(6,'cpb');
individs_model = polyfit(corr_fig3.lat,corr_fig3.da,1);
figure(702); clf;
subplot(311);
plot([50:500] , polyval(individs_model,50:500) , 'k-');
hold on;
scatter(corr_fig3.lat,corr_fig3.da,[ones(1,6)*50 ones(1,6)*100],[1:6 1:6],'filled'); colormap(individs); axis([50 500 0.75e-3 2.25e-3]);
ylabel('init. rew DA (au)'); xlabel('final latency (ms)'); box off;


subplot(312);
good_l = latency(1:numel(stim_list)/2,:);
bad_l = latency(numel(stim_list)/2+1:end,:);
% plot(mean(good_l),'color',[0 174 239]/255); hold on; plot(mean(bad_l),'color',[236,0,145]/255);    
shadedErrorBar([0:160]*5,mean(good_l),std(good_l)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar([0:160]*5,mean(bad_l),std(bad_l)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('Collection latency'); box off;

subplot(313);
good_l = anticip(1:numel(stim_list)/2,:);
bad_l = anticip(numel(stim_list)/2+1:end,:);
% plot(mean(good_l),'color',[0 174 239]/255); hold on; plot(mean(bad_l),'color',[236,0,145]/255);    
shadedErrorBar([0:160]*5,mean(good_l),std(good_l)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar([0:160]*5,mean(bad_l),std(bad_l)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('Anticipatory licking'); box off;
xlabel('Trials');


figure(701); clf;
subplot(141);
good_da = da_early(1:numel(stim_list)/2,:);
bad_da = da_early(numel(stim_list)/2+1:end,:);
% plot(mean(good_da),'color',[0 174 239]/255); hold on; plot(mean(bad_da),'color',[236,0,145]/255);    
shadedErrorBar(-1599:1400,mean(good_da),std(good_da)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar(-1599:1400,mean(bad_da),std(bad_da)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('DA response');

subplot(142);
good_da = da_late(1:numel(stim_list)/2,:);
bad_da = da_late(numel(stim_list)/2+1:end,:);
% plot(mean(good_da),'color',[0 174 239]/255); hold on; plot(mean(bad_da),'color',[236,0,145]/255);    
shadedErrorBar(-1599:1400,mean(good_da),std(good_da)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar(-1599:1400,mean(bad_da),std(bad_da)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('DA response');

subplot(143);
good_l = latency(1:numel(stim_list)/2,:);
bad_l = latency(numel(stim_list)/2+1:end,:);
% plot(mean(good_l),'color',[0 174 239]/255); hold on; plot(mean(bad_l),'color',[236,0,145]/255);    
shadedErrorBar([0:160]*5,mean(good_l),std(good_l)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar([0:160]*5,mean(bad_l),std(bad_l)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('Collection latency'); box off;

subplot(144);
good_l = anticip(1:numel(stim_list)/2,:);
bad_l = anticip(numel(stim_list)/2+1:end,:);
% plot(mean(good_l),'color',[0 174 239]/255); hold on; plot(mean(bad_l),'color',[236,0,145]/255);    
shadedErrorBar([0:160]*5,mean(good_l),std(good_l)./sqrt(numel(stim_list)/2),{'color',[0 174 239]/255}); hold on;
shadedErrorBar([0:160]*5,mean(bad_l),std(bad_l)./sqrt(numel(stim_list)/2),{'color',[236,0,145]/255}); hold on;
ylabel('Anticipatory licking'); box off;

%% run the RNN

simulation_type = 'pavlovian'
% simulation_type = 'pav_unpred'
% simulation_type = 'pav_omit'

switch simulation_type

    case 'pavlovian'
        [net_eval,net] = dlRNN_run_learnDA(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,0);
    case 'pav_omit'
        [net_eval,net] = dlRNN_run_learnDA(net,input_omit,target,act_func_handle,learn_func_handle,transfer_func_handle,0);
    case 'pav_unpred'
        [net_eval,net] = dlRNN_run_learnDA(net,input_unpred,target,act_func_handle,learn_func_handle,transfer_func_handle,0);
    case 'variable_cue'
        [net_eval,net] = dlRNN_run_learnDA(net,input,target,act_func_handle,learn_func_handle,transfer_func_handle,0);
    case 'variable_cue_dist'
%         dist_in{1}(1,:) = input{1}(1,:);
        dist_in{1}(1,:) = zeros(1,3000);
        dist_in{1}(2,:) = zeros(1,3000);
%         dist_in{1}(3,:) = zeros(1,3000);
        dist_in{1}(3,:) = input{1}(3,:);
        dist_in{2}(1,:) = input{2}(1,:);
%         dist_in{2}(1,:) = zeros(1,3000);
        dist_in{2}(2,:) = zeros(1,3000);
        dist_in{2}(3,:) = input{1}(3,:);
        figure(5); imagesc(dist_in{1});
        target_out{1} = target{1}(:,:);
        target_out{2} = target{2}(:,:);
        [net_eval,net] = dlRNN_run_learnDA(net,dist_in,target_out,act_func_handle,learn_func_handle,transfer_func_handle,0);
    otherwise
        
end

% return: net (trained network) and output (end of training simulation)

%% Examine the training process

condition = 1;

figure(800); clf;

for kk=1:numel(output.pass)
    
    if numel(output.pass(kk).chk.v)>0    
%         plot( output.pass(kk).chk.v , ones(1,numel(output.pass(kk).chk.v)) .* kk , 'k.' ); 
        lick_stats.ant(kk) = numel(find(output.pass(kk).chk.v>500 & output.pass(kk).chk.v<1600));
        tmp = find(output.pass(kk).chk.v>1600,1);
        if numel(tmp)>0
            lick_stats.lat(kk) = tmp;
        else
            lick_stats.lat(kk) = 1400;
        end
        hold on;
    end
    
end

[binLick] = TNC_BinAndMean(1:numel(output.pass),lick_stats.ant,10);
[binLag] = TNC_BinAndMean(1:numel(output.pass),lick_stats.lat,10);
figure(800); clf; 
subplot(121);
shadedErrorBar(binLick.bins.center,binLick.bins.avg,binLick.bins.sem);
subplot(122);
shadedErrorBar(binLag.bins.center,binLag.bins.avg,binLag.bins.sem);

plot([100 100],[1 numel(output.pass)],'r--','linewidth',2);
plot([1600 1600],[1 numel(output.pass)],'r--','linewidth',2);
axis tight; box off; set(gca,'YDir','reverse');
xlabel('time');
ylabel('Training reps');
title('Licking output');

for kk=1:numel(output.pass)
        fullmat(kk,:)       = output.pass(kk).chk.o; 
%         fullmatDA(kk,:)   = [0 diff(output.pass(kk).chk.o)]; 
end

figure(801); clf; clear deriv;
imagesc(fullmat);
box off;
xlabel('time');
ylabel('Training reps');
title('Output unit activity');

count = 1;
bin = 100;
clear corr_DA comp_DA total_DA

for kk=bin:bin:size(fullmat,1)
    deriv(count,:) = [0 diff( sgolayfilt( mean(fullmat(kk-bin+1:kk,:),1) , 3 , 101) ) ];
    
    total_DA(count) = trapz(deriv(count,:)) - trapz(deriv(1,:));
    comp_DA(count,1) = trapz(deriv(count,100:500));
    comp_DA(count,2) = trapz(deriv(count,1500:1900));
    
    count = count +1;
end
count = count-1;

corr_DA(1) = 0;
for mm=2:count
    if mm>7
        corr_DA(mm) = corr(comp_DA(mm-7:mm,1),comp_DA(mm-7:mm,2))
    else
        corr_DA(mm) = corr(comp_DA(1:mm,1),comp_DA(1:mm,2))
    end
end

figure(802); clf;
imagesc(deriv);

figure(803);
subplot(121); scatter(1:count,corr_DA,50,'filled'); ylabel('Cue-Reward Correlation'); xlabel(['Training bins (n=' num2str(bin) '/bin)']);
subplot(122); scatter(1:count,total_DA,50,'filled'); ylabel('\Delta Total DA'); xlabel(['Training bins (n=' num2str(bin) '/bin)']);

%% analyze the trained network output by exploring typical sys neuro analysis methods

%% Examine the timeseries cross correlation

%% examine clustered correlations

[cm]        = TNC_CreateRBColormap(1024,'rb');

idx = kmeans(net_eval.cond(3).hr,6)
[vals,inds] = sort(idx);
figure(21); 
subplot(121); imagesc(net_eval.cond(3).hr(inds,:), [-1 1])
subplot(122); imagesc( corr(net_eval.cond(3).hr(inds,:)') , [-1 1])
colormap(cm);

% examine by movement input 
[v_Wi_M, i_Wi_M] = sort(net.wIn(:,9),'descend');

figure(22); 
subplot(221); imagesc(net_eval.cond(1).hr(i_Wi_M,:),[-1 1]); title('AA');
subplot(222); imagesc(net_eval.cond(2).hr(i_Wi_M,:),[-1 1]); title('BB');
subplot(223); imagesc(net_eval.cond(3).hr(i_Wi_M,:),[-1 1]); title('AB');
subplot(224); imagesc(net_eval.cond(4).hr(i_Wi_M,:),[-1 1]); title('BA');
colormap(cm);

figure(22); 
subplot(221); imagesc(net_eval.cond(1).hr(inds,:),[-1 1]); title('AA');
subplot(222); imagesc(net_eval.cond(2).hr(inds,:),[-1 1]); title('BB');
subplot(223); imagesc(net_eval.cond(3).hr(inds,:),[-1 1]); title('AB');
subplot(224); imagesc(net_eval.cond(4).hr(inds,:),[-1 1]); title('BA');
colormap(cm);

%%
figure(23); plot(net_eval.cond(1).hr(:,600),net_eval.cond(4).hr(:,600),'k.'); hold on; plot(net_eval.cond(1).hr(:,200),net_eval.cond(3).hr(:,200),'ro');

%% look at some example cells

% examine interesting hidden units by virtue of their weight onto read-out neuron
[v_Wj_1, i_Wj_1] = sort(net.J(net.oUind(1),:),'descend');
[v_Wj_1a, i_Wj_1a] = sort(net.J(net.oUind(1),:));

[v_Wj_2, i_Wj_2] = sort(net.J(net.oUind(2),:),'descend');
[v_Wj_2a, i_Wj_2a] = sort(net.J(net.oUind(2),:));

% examine by movement input 
[v_Wi_M, i_Wi_M] = sort(net.wIn(:,9),'descend');

fav_ind = randperm(200,8);
fav_ind = [i_Wj_2(1:4) i_Wj_2a(1:4)];
figure(51+1); clf;
% fav_ind = [i_Wj_1(1:4) i_Wj_1a(1:4)];
% figure(51); clf;

% fav_ind = randperm(200,8);
fav_ind = [i_Wi_M(1:4) i_Wi_M(297:300)];
figure(51+2); clf;

for kk = 1:numel(fav_ind)
    for jj=1:numel(net_eval.cond)
        subplot(1,numel(fav_ind),kk);
        plot(net_eval.cond(jj).hr(fav_ind(kk),:),'linewidth',3,'color',cat(jj,:)); hold on;
        axis([0 200 -1 1]); title(num2str(fav_ind(kk)));

    end
end
%% examine PCA of this network
% [R_4_2d, mapping] = compute_mapping(net_eval.cond(3).hr', 'PCA', 2);

average_net = (net_eval.cond(1).hr + net_eval.cond(2).hr + net_eval.cond(3).hr + net_eval.cond(4).hr)./4;

% [mappedA, mapping] = compute_mapping( [net_eval.cond(1).hr , net_eval.cond(2).hr , net_eval.cond(3).hr , net_eval.cond(4).hr]', 'PCA', 3);
[mappedA, mapping] = compute_mapping( average_net', 'PCA', 3);

[cat] = TNC_CreateRBColormap(8,'mapb')

R_1_2d = net_eval.cond(1).hr'*mapping.M;
R_2_2d = net_eval.cond(2).hr'*mapping.M;
R_3_2d = net_eval.cond(3).hr'*mapping.M;
R_4_2d  = net_eval.cond(4).hr'*mapping.M;

figure(24); clf; subplot(1,5,1:3);
plot3(R_1_2d(:,1),R_1_2d(:,2),R_1_2d(:,3),'linewidth',3,'color',cat(1,:)); hold on; 
%     scatter(R_1_2d(1000,1),R_1_2d(1000,2),100,1,'filled'); hold on;
%     scatter(R_1_2d(600,1),R_1_2d(600,2),100,1,'filled'); hold on;
plot3(R_2_2d(:,1),R_2_2d(:,2),R_2_2d(:,3),'linewidth',3,'color',cat(2,:)); hold on; 
%     scatter(R_2_2d(1000,1),R_2_2d(1000,2),100,2,'filled'); hold on;
%     scatter(R_2_2d(600,1),R_2_2d(600,2),100,2,'filled'); hold on;
plot3(R_3_2d(:,1),R_3_2d(:,2),R_3_2d(:,3),'linewidth',3,'color',cat(3,:)); hold on; 
%     scatter(R_3_2d(1000,1),R_3_2d(1000,2),100,3,'filled'); hold on;
%     scatter(R_3_2d(600,1),R_3_2d(600,2),100,3,'filled'); hold on;
plot3(R_4_2d(:,1),R_4_2d(:,2),R_4_2d(:,3),'linewidth',3,'color',cat(4,:)); hold on; 
%     scatter(R_4_2d(1000,1),R_4_2d(1000,2),100,4,'filled'); hold on;
%     scatter(R_4_2d(600,1),R_4_2d(600,2),100,4,'filled'); hold on;

plot(0,0,'.','markersize',50,'color',[0.1 0.1 0.1]); hold on;
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
grid on;

figure(24); subplot(1,5,[4.5,5]); hold off;
for i=1:4
    plot(net_eval.cond(i).hr(1,:),'linewidth',3,'color',cat(i,:)); hold on;
end
ylabel('Shoulder unit (rate; au)');
xlabel('Time');


figure(25); clf;
[dynamics, mapping] = compute_mapping( net_eval.cond(4).hr', 'PCA', 3);

for i=1:2
    plot(dynamics(:,i),'linewidth',3,'color',cat(i,:)); hold on;
end
ylabel('PC Loading');
xlabel('Time');
% figure(4);
% subplot(221); imagesc(R{1}(inds,:),[-1 1])
% subplot(222); imagesc(R{2}(inds,:),[-1 1])
% subplot(223); imagesc(R{3}(inds,:),[-1 1])
% subplot(224); imagesc(R{4}(inds,:),[-1 1])
%% Good way to illustrate properties of the plant?

% Show putative policies across learning, raster plots of licking data for each policy, lick peths, compare to behavioral data?
% 
filter1 = TNC_CreateGaussian(1000,50,2000,1)
colors = [0 0.67 1 ; 0.5 0.5 0.67 ; 1 0.33 0.33]

% filter1(1:100) = 0;
% filter2 = TNC_CreateGaussian(105,10,200,1)./1.11;
% filter2(1:105) = 0;
% filter = filter1-filter2;
% figure(2); clf; plot(filter);
% 
% activity = [zeros(1,300) ones(1,700)];
% figure(3); clf;
% plot(activity); hold on;
% plot(conv(activity,filter,'same'));

figure(4); clf;
reps = 100;
pol_vec = [0.1 0.25 1]

for p=pol_vec
    
    activity = conv([zeros(1,300) ones(1,1300)*p ones(1,1400)*p],filter1,'same');
    
    raster.x = []; raster.y = []; peth = zeros(1,3000);
    for pp=1:reps
        [checks,state] = dlRNN_Pcheck_transfer(activity,50);
        if numel(checks)>0
            raster.y = [raster.y ones(1,numel(checks))*pp];
            raster.x = [raster.x checks];
        end
        peth(raster.x) = peth(raster.x)+1;
    end
    
%     subplot(1,numel([0.1 0.5 0.75]),find(p==[0.1 0.5 0.75])); plot(raster.x,raster.y,'k.'); hold on; plot(100*conv(peth./reps,[0 ones(1,60)./60 0],'same')-105); axis([0 3000 -105 reps+1]); plot(activity*100-105); box off;
    subplot(312); 
        plot(raster.x,raster.y+reps*(find(p==pol_vec)-1),'k.','color',colors(find(p==pol_vec),:)); hold on;
        axis([0 2.5e3 0 reps*3]); box off; axis off;
    subplot(313)
        plot(conv(peth./reps,[0 ones(1,80)./80 0],'same'),'color',colors(find(p==pol_vec),:)); hold on; box off;
        axis([0 2.5e3 0 0.5]); plot([1600 1600],[0 1],'k-');
    subplot(311)
        plot(activity,'color',colors(find(p==pol_vec),:)); hold on;box off;
        axis([0 2.5e3 0 1]);
    
end