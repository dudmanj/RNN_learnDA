%% LearnDA simulations scripts
global pt_on;
pt_on = 0;

% Code to run simulation and display key output measures:

stim_list = [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 ];
inits = repmat(ii([141 159 166 176 191 150]),1,6);

clear run;
parfor g = 1:numel(stim_list)

    net_init = gens.dets(inits(g)).net % diverse initial states
                
    net_init.wIn(net.oUind,:) = [0 0];
    tau_trans = 1; % now controls wJ learning rate
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

%% Summary display plot for talks of training experience.

% To plot:

% Trialwise / sessionwise:
% Anticipatory licking
% Fraction lick+ trials
%     per trial lick analysis is here:
%     run(1).output.pass(1).chk.v

% cued vs uncued latency
%     run(1).output.pass(1).lat
%     run(1).output.pass(1).lat_u
%     run(1).output.pass(1).lat_o
clear summary_data

[stim_map] = [1 0 0.67 ; 0 1 0.67 ; 0 0.67 1];
lk_kern = TNC_CreateGaussian(500,40,1000,1);
learn_ranges = [1 40 120; 20 60 160];
learn_ranges_da = [1 61; 60 120];

jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
s_scl = 2;
m_scl = 1;
cnt = 1;

% close all;

for g=[1 13 25]

    switch stim_list(g)
        case -1
            figure('Name',['Init ' num2str(inits(g)) ' : StimLick-' ],'NumberTitle','off'); clf;
        case 0
            figure('Name',['Init ' num2str(inits(g)) ' : Cntrl' ],'NumberTitle','off'); clf;
        case 1
            figure('Name',['Init ' num2str(inits(g)) ' : StimLick+' ],'NumberTitle','off'); clf;
    end

    summary_data.cond(g) = stim_list(g);
    summary_data.init(g) = inits(g);
    summary_data.net(g).neti = gens.dets(inits(g)).net;
    
    % For each pass generate the continuous lick and dopamine traces
    summary_data.trials(g).tr = run(g).net.update:run(g).net.update:numel(run(g).output.pass);
    summary_data.lik(g).lk = zeros(numel(summary_data.trials(g).tr),size(run(g).output.cond.out,2));
    summary_data.lik(g).lks = zeros(numel(summary_data.trials(g).tr),size(run(g).output.cond.out,2));
    summary_data.lat(g).la = zeros(numel(summary_data.trials(g).tr),3);
    summary_data.lat(g).las = zeros(numel(summary_data.trials(g).tr),3);

%     for gg=1:numel(run(g).output.pass)
    cnt=1;
    for gg=run(g).net.update:run(g).net.update:numel(run(g).output.pass)
        summary_data.lat(g).la(cnt,:) = [run(g).output.pass(gg).lat run(g).output.pass(gg).lat_u run(g).output.pass(gg).lat_o];
        summary_data.lik(g).lk(cnt,unique(run(g).output.pass(gg).chk.v)) = 1;
        summary_data.lik(g).lks(cnt,:) = conv(summary_data.lik(g).lk(cnt,:),lk_kern,'same');
        cnt = cnt+1;
    end

    for h=1:3
%         summary_data.lat(g).las(:,h) = conv(summary_data.lat(g).la(:,h),[0 ones(1,50) 0]/50,'same');
        summary_data.lat(g).las(:,h) = sgolayfilt(summary_data.lat(g).la(:,h),3,11);
    end

    subplot(141); imagesc(summary_data.lik(g).lks); colormap(bone); ylabel('Trials');
    subplot(142); plot(summary_data.trials(g).tr,summary_data.lat(g).las(:,1),'color',stim_map(3,:)); hold on;
    axis([0 800 0 750]);
    plot(summary_data.trials(g).tr,summary_data.lat(g).las(:,2),'color',stim_map(1,:));  
    ylabel('Latency');  xlabel('Trials');
    for h=1:3
        subplot(143); plot(1000*mean(summary_data.lik(g).lks(learn_ranges(1,h):learn_ranges(2,h),:)),'color',stim_map(h,:)); hold on;
    end
    axis([0 3000 -1 7]);
    ylabel('Lick rate (Hz)');  legend({'early' 'mid' 'late'},'location','northwest'); xlabel('Time');


    num_das = size(run(g).pred_da_move,1);
    summary_data.da(g).c = zeros(num_das,size(run(g).output.cond.out,2));
    summary_data.da(g).u = zeros(num_das,size(run(g).output.cond.out,2));
    summary_data.da(g).o = zeros(num_das,size(run(g).output.cond.out,2));

    for ggg=1:num_das
        summary_data.da(g).c(ggg,:) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
    end

    for h=1:size(learn_ranges_da,2)
        subplot(144); plot(1000*mean(summary_data.da(g).c(learn_ranges_da(1,h):learn_ranges_da(2,h),:)),'color',stim_map(h,:)); hold on;
    end
    axis([0 3000 -1 7]);
    ylabel('DA resp. (a.u.)');  xlabel('Time'); legend({'1-300' '300-600'},'location','northwest');

end

%% Comparison across distinct run types

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

figure(602); clf;
    shadedErrorBar( [1:200]*run(1).net.update  , mean(all_latency,1) , std(all_latency,[],1)./sqrt(size(all_latency,1)) , {'color',[1 0.1 0.2]}); hold on;
    shadedErrorBar( [1:200]*run(1).net.update , mean(all_latency_u,1) , std(all_latency_u,[],1)./sqrt(size(all_latency_u,1)) ); hold on;
    ylabel('Latency to collect reward (ms)'); xlabel('Training trial bins');
    axis([0 200 0 500]);
