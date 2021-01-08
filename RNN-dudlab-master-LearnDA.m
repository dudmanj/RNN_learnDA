%% RNN-dudlab-master_learnDA
% High level script to control the construction and training of RNNs
 
% Current ideas about future development:
% 1) DA transients would be estimated with licking state transition and sensory responses. Try computing these form the network
% 2) Would like to be able to modulate the learning rate proportional to DA transients
% 3) Would like to actually drive a pair of outputs that control the forward and reverse rate on licking 
% 4) Forward rate (put D1) would only have learning rate increased for positive DA transients and Reverse rate (put D2) only for decreases
% 5) Another idea is that there is a learning rate output unit trained on the network. Not sure what objective to use for this, but an interesting idea.

%% theoretical minimum error
lick_k = TNC_CreateGaussian(200,5,400,1);
% lick_k(1:200)=0;
its = 200;
l_its_max=4;
delay = zeros(l_its_max,its);
delay_u = zeros(l_its_max,its);
cost = zeros(1,its);
licks = zeros(its,3000);
figure(21); clf;
for l_its=1:l_its_max
    licks = zeros(its,3000);
    for kk=1:its

        put_policy = [zeros(1,100) ones(1,2400) zeros(1,500)].*(0*(l_its-1))+[zeros(1,100) ones(1,10).*0 zeros(1,1590) ones(1,10).*0.05.*(l_its-1).^2 zeros(1,1290)];        
        outputs_t = dlRNN_Pcheck_transfer(put_policy);
        tmp = find(outputs_t>1600,1);
        delay_u(l_its,kk) = outputs_t(tmp)-1600;

        put_policy = [zeros(1,500) ones(1,2000) zeros(1,500)].*(0.0004*(l_its-1).^2)+[zeros(1,100) ones(1,10).*0.005.*(l_its-1).^2 zeros(1,1590) ones(1,10).*0.05.*(l_its-1).^2 zeros(1,1290)];        
        outputs_t = dlRNN_Pcheck_transfer(put_policy);
        tmp = find(outputs_t>1600,1);
        delay(l_its,kk) = outputs_t(tmp)-1600;
        cost(l_its,kk) = 500 * (  1-exp(-delay(kk)/500) );
        licks(its,outputs_t) = 1;
        licks(its,:) = conv(licks(its,:),lick_k,'same');

    end
%     subplot(4,1,l_its);

    subplot(2,3,1:2); plot(mean(licks)); hold on; xlabel('Time (ms)'); ylabel('Licking'); box off; legend; axis([ 0 3000 0 0.005]);
    subplot(2,3,4:5); plot(put_policy); hold on; xlabel('Time (ms)'); ylabel('Policy'); box off; axis([ 0 3000 0 0.01]);
    subplot(2,3,[3 6]); hold off; plot(1:l_its,mean(delay(1:l_its,:),2),'r'); hold on; plot(1:l_its,mean(delay_u(1:l_its,:),2),'k'); legend('Cued','Uncued'); xlabel('Learning stage'); ylabel('Latency (ms)'); box off;
    
    
end
% mean(cost)
% std(cost)
% mean(delay)

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
actLogic = 0;
    lognorm    = 1;     % Use log normal distribution input weights


% Reasonable default values
N   = 50;              % RNN Units
B   = size(target{1},1);% Outputs
I   = size(input{1},1); % Inputs
p   = 1;                % Sparsity
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
g_vec = [1.3];
% tau_vec = [5 10 20 25 50];
tau_vec = [20];

for jj=1:generations
    
%     net.N   = N_vec(randperm(5,1));
    net.g   = g_vec(randperm(numel(g_vec),1));
    net.tau = tau_vec(randperm(numel(tau_vec),1));    
    
    J = zeros(net.N,net.N);
    for i = 1:net.N
        for j = 1:net.N
            if rand <= p
                J(i,j) = net.g * randn / sqrt(p*net.N);
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

figure(9); subplot(1,6,1:3); imagesc(evol.out(ii,:),[-1 1]); colormap(cm); ylabel('Generations');  xlabel('Time'); box off;
subplot(1,6,4); plot(evol.lag(ii),1:generations,'k.'); set(gca,'YDir','reverse'); box off; xlabel('Collect latency');
subplot(1,6,5); semilogx(evol.g(ii),1:generations,'k.'); set(gca,'YDir','reverse'); box off;  xlabel('g');
subplot(1,6,6); plot(abs(evol.eig(ii)),1:generations,'k.'); set(gca,'YDir','reverse'); box off;  xlabel('eig'); set(gca,'TickDir','out');


figure(8); clf;
subplot(2,2,1); plot(evol.g,evol.err,'k+'); box off; xlabel('g'); ylabel('error');
% subplot(2,2,2); plot(tmp_g,tmp_tau,'ko'); 
subplot(2,2,4); semilogx(evol.tau,evol.err,'k+'); box off; xlabel('tau'); ylabel('error'); 

%----------------------------------------------
% Find median initialized network
%----------------------------------------------

% take best 500 ms tau network
% index = find(tmp > median(tmp(tmp_g==2)),1);
index = ii(1);
net = gens.dets(index).net;
figure(10); clf; subplot(211); plot(gens.dets(index).out); title('Optimal g=2 evolved '); box off;ylabel('Output unit');
gens.dets(index).err

% take best 15 ms tau network
% index = find(tmp > median(tmp),1);
% index = find(tmp == min(tmp(tmp_tau==2000)),1);
% index = find(tmp == min(tmp(tmp_g==1.3)),1);
% index=ii(50);
% index = find(tmp > 750 & tmp < 1000 & tmpL > 1000 & tmpL < 1500,1);
index=ii(194)
gens.dets(index).net.g

% index = ii(round(generations./5));
net = gens.dets(index).net;
net_init = gens.dets(index).net;
% gens.dets(index).err
figure(10); subplot(212); plot(gens.dets(index).out); title('Median tau=2000ms evolved ');  xlabel('Time'); ylabel('Output unit'); box off;

% net.wIn(net.oUind,:) = 0;

figure(11); clf; plot(eig(gens.dets(ii(1)).net.J),'o'); hold on; plot(eig(gens.dets(index).net.J),'o');

%% Compute the empirical anticipatory licking cost function
emp_ant_cost = polyfit(evol.ant_lck,evol.err_ant,2) 

bin_dat = TNC_BinAndMean(evol.ant_lck,evol.err_ant,7);

figure(2); clf; 
plot(evol.ant_lck,evol.err_ant,'k.','color',[0.5 0.5 0.5 0.5]);
hold on;
plot(bin_dat.bins.center,bin_dat.bins.avg,'ko','MarkerSize',10,'MarkerFace','k');
plot(0:0.1:9,polyval(emp_ant_cost,0:0.1:9),'r-');

%% train the RNN

switch simulation_type
   
    case 'pavlovian'

        clear run;
        figure(1); clf;
        parfor g = 1:24

            net_init = gens.dets(index).net % diverse initial states
            net_init.wIn(net.oUind,:) = [0 0];
            tau_trans = g;
            [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans);

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

[err_map] = TNC_CreateRBColormap(numel(run),'cpb'); 
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
    plot(sgolayfilt(model(g).anticip,3,21),sgolayfilt(model(g).latency,3,21),'color',err_map(g,:)./2); hold on;
    
    figure(502); subplot(4,6,g); imagesc(run(g).pred_da_move+100*run(g).pred_da_sense);

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

figure(501);
    plot(sgolayfilt(model_runs.anticip_d,3,21) , sgolayfilt(model_runs.latency_d,3,21) ,'w', 'linewidth', 3 ); hold on;
            title('Cost surface'); ylabel('Latency (ms)'); xlabel('Anticipatory licks');

all_latency = zeros(numel(run) , 200); 
all_latency_u = zeros(numel(run) , 200); 

for kk=1:numel(run)
    all_latency(kk,1:numel(model(kk).latency)) = model(kk).latency(1,1:numel(model(kk).latency));
    all_latency_u(kk,1:numel(model(kk).latency_u)) = model(kk).latency_u(1,1:numel(model(kk).latency_u));
end

figure(503); clf;
    shadedErrorBar( [1:200]*run(1).net.update  , mean(all_latency,1) , std(all_latency,[],1)./sqrt(size(all_latency,1)) , {'color',[1 0.1 0.2]}); hold on;
    shadedErrorBar( [1:200]*run(1).net.update , mean(all_latency_u,1) , std(all_latency_u,[],1)./sqrt(size(all_latency_u,1)) ); hold on;
            ylabel('Latency to collect reward (ms)'); xlabel('Training trial bins');
%             legend('Cued','Uncued');
axis([0 200 0 1500]);

figure(500); boxplot(trials_to_criterion); ylabel('Trials to criterion');





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
