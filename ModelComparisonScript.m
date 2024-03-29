
load ~/'Dropbox (HHMI)'/matlab-common-init.mat
% load ~/'Dropbox (HHMI)'/run.mat
% load ~/'Dropbox (HHMI)'/cued-da-run.mat

%% LearnDA simulations scripts for comparing Cntrl (various inits) w/ StimLick+ StimLick- Stim+Lick+
global pt_on;
pt_on = 1;

% Code to run simulation and display key output measures:
num_sims = 24;
stim_list = zeros(1,num_sims);

% inits = repmat(ii([201 132 110 118 141 195]),1,4);
inits = repmat(ii([201 130 98 119 141 195]),1,4);
tmp = repmat([1 1.25 1.5 1.75],6,1);
wIn_vec = tmp(1:numel(tmp));
% tau_vec = ones(1,num_sims)*2;
tau_vec = tmp(1:numel(tmp))+1;
sat_vec = [randperm(6) randperm(6) randperm(6) randperm(6)]+4;

clear run;

parfor g = 1:numel(stim_list)

    net_init = gens.dets(inits(g)).net % diverse initial states
                
    net_init.wIn(net.oUind,:) = [0 wIn_vec(g)];
    tau_trans = tau_vec(g); % now controls eta_wIn learning rate
    filt_scale = 50; % plant scale factor currently
    trans_sat = sat_vec(g);
    stim = stim_list(g);

    [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u,pred_da_move_o,pred_da_sense_o] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale,trans_sat);

    run(g).output = output;
    run(g).net = net_out;
    run(g).pred_da_sense = pred_da_sense;
    run(g).pred_da_move = pred_da_move;
    run(g).pred_da_sense_u = pred_da_sense_u;
    run(g).pred_da_move_u = pred_da_move_u;
    run(g).pred_da_sense_o = pred_da_sense_o;
    run(g).pred_da_move_o = pred_da_move_o;
    disp(['Completed run: ' num2str(g)]);

end

save ~/'Dropbox (HHMI)'/run-ctrl-PEstim run stim_list inits
% save ~/'Dropbox (HHMI)'/run-ctrl-noC-good-PhasicKO run stim_list inits
% save ~/'Dropbox (HHMI)'/run-ctrl-noC-good-DADeplete run stim_list inits
% save ~/'Dropbox (HHMI)'/run-ctrl-noC-good-WithBeta run stim_list inits
% save ~/'Dropbox (HHMI)'/run-ctrl-noC-good-DaIsPE run stim_list inits
% save ~/_PROJECTS/Luke-LearnDA/run-ctrl run stim_list inits

%% LAST BITS NEEDED FOR PAPER FIGURES
% 1. Plot Cost vs Ant vs React for all Cntrl model sims X
% 2. Cued-DA stim DA and licking predictions X
% 3. RPE predictions X
% 4. Lick+ / Lick- predictions
% 5. PE|lick+ and PE|lick- vs training trials X
% 6. DA and Licking predictions for stimLick-, stimLick+, Stim++Lick+

%% 0. Need to compute a function that converts policy into a "transient" variable equivalent to experimental data

trans_vec = 0:0.5:11;
lat_vec = zeros(size(trans_vec))
reps=1:1000;
lat = zeros(size(reps));
activity = zeros(1,3000);

for j=1:numel(trans_vec)
    activity(1600:1610) = trans_vec(j);
    for i=reps
        [checks,state] = dlRNN_Pcheck_transfer(activity,50);
        tmp = checks(checks>1600);
        if numel(tmp>0)
            lat(i) = tmp(1)-1600;
        else
            lat(i) = 1400;
        end
    end
    lat_vec(j) = mean(lat);
end        

figure(10); clf;
plot(trans_vec,lat_vec);
rct_off = lat_vec(end);
react_model = fit( trans_vec' , lat_vec'-rct_off,'exp1');    
react_sm = react_model(trans_vec)+rct_off;
hold on; plot(trans_vec,react_sm);

%% 0.5 Compute full error surface
s_map = TNC_CreateRBColormap(1000,'cpb');
trans_vec = 0:0.5:11;
sust_vec = -0.25:0.05:1;
reps=1:50;
lat = zeros(numel(sust_vec),numel(reps));
lat_mat = zeros(numel(sust_vec),numel(trans_vec));
cost_mat = zeros(numel(sust_vec),numel(trans_vec));

for ss=1:numel(sust_vec)
    for tt=1:numel(trans_vec)

        activity = [zeros(1,600) ones(1,950).*sust_vec(ss) zeros(1,1450)];
        activity(1600:1610) = trans_vec(tt);

        for i=reps
            [checks,state] = dlRNN_Pcheck_transfer(activity,50);
            tmp = checks(checks>1600);
            if numel(tmp)>0
                lat(ss,i) = tmp(1)-1600;
            else
                lat(ss,i) = 1400;
            end
        end
        lat_mat(ss,tt) = mean(lat(ss,:));
        cost_mat(ss,tt) = 1-exp(-lat_mat(ss,tt)/500);
    end
%     cost_mat(ss,:) = sgolayfilt(cost_mat(ss,:),3,11);
end

% for tt=1:numel(trans_vec)
%     cost_mat(:,tt) = sgolayfilt(cost_mat(:,tt),3,11);
% end

trans_axis = [react_sm']-100;
sust_axis = sust_vec*8;
% sust_axis = sust_axis-min(sust_axis);

x = reshape(sust_axis'*ones(size(trans_axis)),numel(cost_mat),1);
y = reshape(ones(size(sust_axis))'*trans_axis,numel(cost_mat),1);
z = reshape(cost_mat,numel(cost_mat),1);

cost_surf_sims = fit([x y],z,'poly33');

figure(100); clf;

set(0,'DefaultFigureRenderer','painters');
s = surf(sust_axis,trans_axis,cost_mat'); hold on;
s = plot(cost_surf_sims);

s.EdgeColor = [0.35 0.35 0.35];
s.FaceAlpha = 0.85;
s.FaceLighting = 'gouraud';
colormap(s_map);
xlabel('Anticipatory');
ylabel('Reactive');
zlabel('Cost');
% axis([0 3 0 275 0 1.2]);
view(48,30);
% set(gca,'YDir','reverse');
set(gca,'Color',[0.95 0.95 0.95]);
box on; 

%% 1. Plot Cost vs Ant vs React for all Cntrl model sims
clear summary_data
cmap = TNC_CreateRBColormap(6,'mapb');
cmap2 = repmat(cmap(1:6,:),3,1);
fOff = 200; % for when rand wIn init is used
figure(1+fOff); clf;
figure(2+fOff); clf;
figure(2); clf;

scnt = 1;
% [vals,inds] = sort(tau_vec+sat_vec);
inds = 1:num_sims;

for g=1:numel(run)

    if stim_list(g)==0 % Only use Cntrl simulations

        cnt = 1;
        for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

            tmp_lat = find(run(g).output.pass(gg).chk.v > 1600,1,'first');
            if numel(tmp_lat)==1
                summary_data.analysis(1).lat(scnt,cnt) = run(g).output.pass(gg).chk.v(tmp_lat)-1600;
            else
                summary_data.analysis(1).lat(scnt,cnt) = 1400;
            end
            summary_data.analysis(1).cost(scnt,cnt)     = 1-exp(-summary_data.analysis(1).lat(scnt,cnt)/500);
            summary_data.analysis(1).rct(scnt,cnt)      = react_model(run(g).output.pass(gg).chk.npi(1605))+50;
%             summary_data.analysis(1).ant(scnt,cnt)      = numel(find(run(g).output.pass(gg).chk.v > 1100 & run(g).output.pass(gg).chk.v > 1600));
            summary_data.analysis(1).outs(scnt,:,cnt)   = run(g).output.pass(gg).chk.o;
            summary_data.analysis(1).ant(scnt,cnt) = run(g).output.pass(gg).anticip;

            label_txt{scnt} = [num2str(inits(g)) '-' num2str(tau_vec(g),2) '-' num2str(wIn_vec(g),1)];

            cnt = cnt+1;

        end

% METHOD FOR COMPUTING LEARNING TRAJECTORIES
        summary_data.analysis(1).ant(scnt,:) = sgolayfilt(summary_data.analysis(1).ant(scnt,:),3,11);
        trial_rng = 140:160; % (i.e. 600 - 800)

        % Compute offset for fitting exponential
        trans_off = mean(summary_data.analysis(1).rct(scnt,trial_rng));
        sust_off = mean(summary_data.analysis(1).ant(scnt,trial_rng));
        lat_off = mean(summary_data.analysis(1).lat(scnt,trial_rng));
        
        % Fit an exponential to the data
        trial_rng = 1:161;
        sustained_model = fit( trial_rng' , summary_data.analysis(1).ant(scnt,:)'-sust_off,'exp2','Lower',[-6 -0.1 -6 -0.1],'Upper',[0 +0.05 0 +0.05]);    
        transient_model = fit( trial_rng' , summary_data.analysis(1).rct(scnt,:)'-trans_off,'exp2','Lower',[50 -0.05 50 -0.5],'Upper',[500 0 500 0]);
        latency_model = fit( trial_rng' , summary_data.analysis(1).lat(scnt,:)'-lat_off,'exp2','Lower',[-5 -0.05 -5 -0.5],'Upper',[1000 0 1000 0]);
        
        transient_sm = transient_model(trial_rng)+trans_off;
        sustained_sm = sustained_model(trial_rng)+sust_off;
        latency_sm = latency_model(trial_rng)+lat_off;
        
        figure(2);
        total_sims = sum(stim_list==0);
%         subplot(total_sims,3,((find(inds==scnt)-1)*3)+1); plot(summary_data.analysis(1).rct(scnt,:)); hold on; plot(transient_sm,'linewidth',3); axis([0 41 0 500]); axis off;
%         subplot(total_sims,3,((find(inds==scnt)-1)*3)+2); plot(summary_data.analysis(1).ant(scnt,:)); hold on; plot(sustained_sm,'linewidth',3); axis([0 161 0 8]); axis off;
%         subplot(total_sims,3,((find(inds==scnt)-1)*3)+3); plot(summary_data.analysis(1).lat(scnt,:)); hold on; plot(latency_sm,'linewidth',3); axis tight; axis off;
        subplot(1,3,1); plot(transient_sm,'linewidth',find(wIn_vec(g)==unique(wIn_vec)),'color',cmap(find(inits(g)==inits(1:6)),:)); axis([0 41 50 400]); hold on;
        subplot(1,3,2); plot(sustained_sm,'linewidth',find(wIn_vec(g)==unique(wIn_vec)),'color',cmap(find(inits(g)==inits(1:6)),:)); axis([0 161 0 8]); hold on;
        subplot(1,3,3); plot(log10(latency_sm),'linewidth',find(wIn_vec(g)==unique(wIn_vec)),'color',cmap(find(inits(g)==inits(1:6)),:)); axis([0 161 1.8 3.2]); hold on;
        
        summary_data.analysis(1).final_lat(scnt) = lat_off
        summary_data.analysis(1).init_id(scnt) = find(inits(g)==unique(inits));

        figure(2+fOff);
        plot3(sustained_sm,transient_sm,log10(latency_sm),'k-','linewidth',1.5); hold on;
            view(48,30);
            % set(gca,'YDir','reverse');
            set(gca,'Color',[0.95 0.95 0.95]);
            box on; 
            axis([0 6 75 350 1.75 3.1]);

        figure(1+fOff);
        if scnt==1    
            set(0,'DefaultFigureRenderer','painters');
            s = plot(cost_surf_sims); hold on;
            s.EdgeColor = [0.35 0.35 0.35];
            s.FaceAlpha = 0.85;
            s.FaceLighting = 'gouraud';
            colormap(s_map);
            xlabel('Anticipatory');
            ylabel('Reactive');
            zlabel('Cost');
            % axis([0 3 0 275 0 1.2]);
            view(48,30);
            % set(gca,'YDir','reverse');
            set(gca,'Color',[0.95 0.95 0.95]);
            box on; 
        end
            plot3(sustained_sm,transient_sm,cost_surf_sims(sustained_sm,transient_sm+200),'w-','linewidth',1); hold on;
            box on; view(48,30);
            set(gca,'Color',[0.95 0.95 0.95]);
            grid on;
            ylabel('Reactive');
            xlabel('Anticipatory');
            zlabel('Cost');
            axis([-2 6 100 750 0.1 1]);
            drawnow;


        scnt = scnt + 1;
    end

end

% legend(label_txt);
% figure(); plot3(summary_data.analysis(1).lat(1,:),summary_data.analysis(1).ant(1,:),summary_data.analysis(1).cost(1,:));

%% 2. Cued-DA stim simulation experiments in ACTR
global pt_on;
pt_on = 1;

num_sims = 12;
inits = repmat(ii([201 130 98 119 141 195]),1,2);
sat_vec = repmat(randperm(6)+4,1,2);
stim_list = [ones(1,6) 20*ones(1,6)];
wIn_vec = zeros(1,num_sims);
tau_vec = ones(1,num_sims);

clear run;

parfor g = 1:numel(stim_list)

    net_init = gens.dets(inits(g)).net % diverse initial states
                
    net_init.wIn(net.oUind,:) = [0 0];
    tau_trans = tau_vec(g); % now controls eta_wIn learning rate
    filt_scale = 50; % plant scale factor currently
    trans_sat = sat_vec(g);
    stim = stim_list(g);

    [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u,pred_da_move_o,pred_da_sense_o] = dlRNN_train_cuedDAstim(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale,trans_sat);

    run(g).output = output;
    run(g).net = net_out;
    run(g).pred_da_sense = pred_da_sense;
    run(g).pred_da_move = pred_da_move;
    run(g).pred_da_sense_u = pred_da_sense_u;
    run(g).pred_da_move_u = pred_da_move_u;
    run(g).pred_da_sense_o = pred_da_sense_o;
    run(g).pred_da_move_o = pred_da_move_o;
    disp(['Completed run: ' num2str(g)]);

end

save ~/'Dropbox (HHMI)'/cued-da-run2 run stim_list inits

%% 2.5 Plot figure panels for cued-DA predictions

[stim_map] = [1 0 0.67 ; 0 1 0.67 ; 0 0.67 1];
lk_kern = TNC_CreateGaussian(500,40,1000,1);
jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
s_scl = 2;
m_scl = 1;
cnt = 1;
sl_cnt = 1;

for g=1:numel(run)

    switch stim_list(g)
        case 0
            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                summary_data.analysis(2).da.c(ggg,:,cnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
            end
            cnt = cnt+1;

        case 20
            num_das = size(run(g).pred_da_move,1);        
            for ggg=1:num_das
                summary_data.analysis(2).da_sl.c(ggg,:,sl_cnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
            end
            sl_cnt = sl_cnt+1;
            
    end

end

% Assumes that ValuModelSims code has been run and sim_summary data
% structure is availble for plotting


colors = [0 0.67 1 0.5; 0 0 0 0.5; 1 0 0.67 0.5];

% plot magnitude of cue RPE over trials as a function of stim intensity
figure(12); clf;
tmp_actr_cal = mean(summary_data.analysis(2).da.c(:,:,:),3);
tmp_actr_sup = mean(summary_data.analysis(2).da_sl.c(:,:,:),3);
plot([0 800],[0 0],'k-'); hold on;

for sims=[1 3]
    plot(100:100:700,sim_summary.rpe(100:100:700,critic.cueTime,sims),'o--','linewidth',3,'color',colors(sims,:)); hold on;
    if sims==1
        plot(100:100:700,tmp_actr_cal(21:20:141,276)*1000,'o-','linewidth',3,'color',colors(sims,:));        
    elseif sims==3
        plot(100:100:700,tmp_actr_sup(21:20:141,276)*1000,'o-','linewidth',3,'color',colors(sims,:));        
    end
end

xlabel('Trials'); ylabel('Cued DA magnitude'); title('Cued DA stim predictions');
box off; legend({'','TD Cal','ACTR Cal','TD Supra','ACTR Supra'},'location','northwest');
axis([50 750 -0.5 3]);

% Now plot the predicted DA transients
figure(13); clf;
subplot(211); % calibrated prediction
td_da_predict = zeros(1,3000);
val_func = [0 diff(sim_summary.critic(1).v)];
td_da_predict(100:100:2500) = val_func(6:end);
td_da_predict = conv(td_da_predict,kern,'same');
plot(-1599:1400,td_da_predict,'--','color',colors(1,:),'linewidth',2); hold on;
plot(-1599:1400,mean(tmp_actr_cal(120:160,:)),'-','color',colors(1,:),'linewidth',2); hold on;
axis([-1500 1500 -2.5e-3 2.5e-3]); box off;

subplot(212); % supra prediction
td_da_predict = zeros(1,3000);
val_func = [0 diff(sim_summary.critic(2).v)];
td_da_predict(100:100:2500) = val_func(6:end);
td_da_predict = conv(td_da_predict,kern,'same');
plot(-1599:1400,td_da_predict,'--','color',colors(3,:),'linewidth',2); hold on;
plot(-1599:1400,mean(tmp_actr_sup(120:160,:)),'-','color',colors(3,:),'linewidth',2); hold on;
axis([-1500 1500 -2.5e-3 2.5e-3]); box off;


%% 3. RPE predictions
clear summary_data;
[stim_map] = [1 0 0.67 ; 0 1 0.67 ; 0 0.67 1];
lk_kern = TNC_CreateGaussian(500,40,1000,1);
jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
s_scl = 2;
m_scl = 1;
cnt = 1;

cue_win = 100:600;
rew_win = 1600:2100;

for g=1:numel(run)

    if stim_list(g)==0
            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                summary_data.analysis(3).da.c(ggg,:,cnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                summary_data.analysis(3).da.u(ggg,:,cnt) = conv( s_scl*run(g).pred_da_sense_u(ggg,:) + m_scl*run(g).pred_da_move_u(ggg,:) , kern , 'same');
                summary_data.analysis(3).da.o(ggg,:,cnt) = conv( s_scl*run(g).pred_da_sense_o(ggg,:) + m_scl*run(g).pred_da_move_o(ggg,:) , kern , 'same');

                summary_data.analysis(3).DA_resp.c_cue_int(ggg,cnt) = trapz( summary_data.analysis(3).da.c(ggg,cue_win,cnt) );
                summary_data.analysis(3).DA_resp.c_rew_int(ggg,cnt) = trapz( summary_data.analysis(3).da.c(ggg,rew_win,cnt) );
                summary_data.analysis(3).DA_resp.u_rew_int(ggg,cnt) = trapz( summary_data.analysis(3).da.u(ggg,rew_win,cnt) );
                summary_data.analysis(3).DA_resp.o_rew_int(ggg,cnt) = trapz( summary_data.analysis(3).da.o(ggg,rew_win,cnt) );

            end
            cnt = cnt+1;
    end

end

% 3.5 Plotting RPE comparisons

% init_choice = randperm(18,9);
init_choice = 1:numel(find(stim_list==0));
% init_choice = [3:6:21];

for qq=1:7 % hundred trial bins
    
    curr_bin=[-10:10] + (qq*20);

    summary_data.analysis(3).DA_resp.c_cue_bin(qq,:) = mean( summary_data.analysis(3).DA_resp.c_cue_int(curr_bin,init_choice),1 );
    summary_data.analysis(3).DA_resp.c_rew_bin(qq,:) = mean( summary_data.analysis(3).DA_resp.c_rew_int(curr_bin,init_choice),1 );
    summary_data.analysis(3).DA_resp.u_rew_bin(qq,:) = mean( summary_data.analysis(3).DA_resp.u_rew_int(curr_bin,init_choice),1 );
    summary_data.analysis(3).DA_resp.o_rew_bin(qq,:) = mean( summary_data.analysis(3).DA_resp.o_rew_int(curr_bin,init_choice),1 );
    
end

figure(32); clf;
subplot(121);
plot([0 800],[0 0],'k--'); hold on;
errorbar(0:100:700,[mean(mean(summary_data.analysis(3).DA_resp.c_cue_int(5:10,:),1)) mean(summary_data.analysis(3).DA_resp.c_cue_bin,2)'],[std(mean(summary_data.analysis(3).DA_resp.c_cue_int(5:10,:),1),[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2)) std(summary_data.analysis(3).DA_resp.c_cue_bin,[],2)'./sqrt(numel(init_choice))],'ro-','linewidth',3);
axis([0 800 -0.2 1.2]);
box off; xlabel('Training trials'); ylabel('Simulated cued DA resp. (au)')
subplot(122);
plot([0 810],[0 0],'k-.'); hold on;
% errorbar(100:100:800,mean(summary_data.analysis(3).DA_resp.c_rew_bin,2),std(summary_data.analysis(3).DA_resp.c_rew_bin,[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2)),'ro-','linewidth',3); hold on;
% errorbar(100:100:800,mean(summary_data.analysis(3).DA_resp.u_rew_bin,2),std(summary_data.analysis(3).DA_resp.u_rew_bin,[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2)),'ko-','linewidth',3); hold on;
errorbar(100:100:700,mean(summary_data.analysis(3).DA_resp.c_rew_bin,2),std(summary_data.analysis(3).DA_resp.c_rew_bin,[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2)),'ro-','linewidth',3); hold on;
errorbar(100:100:700,mean(summary_data.analysis(3).DA_resp.u_rew_bin,2),std(summary_data.analysis(3).DA_resp.u_rew_bin,[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2)),'ko-','linewidth',3); hold on;
xxx = mean(summary_data.analysis(3).DA_resp.o_rew_bin,2);
% yyy = std(summary_data.analysis(3).DA_resp.o_rew_bin,[],2)./sqrt(size(summary_data.analysis(3).DA_resp.c_rew_bin,2));
yyy = std(summary_data.analysis(3).DA_resp.o_rew_bin,[],2);
errorbar(100:100:700,xxx(1:7),yyy(1:7),'bo-','linewidth',3); hold on;
axis([0 810 -0.5 5]); 
box off; xlabel('Training trials'); ylabel('Simulated reward DA resp. (au)')

% Compute reward responses for cntrl, uncued, omit same 100 trial bins
% Show RPE traces for stable / end of learning

figure(31); clf;

stable_trial_range = 140:160;

summary_data.analysis(3).DA_resp.c_avg = mean(mean(summary_data.analysis(3).da.c(stable_trial_range,:,:),1),3);
summary_data.analysis(3).DA_resp.c_sem = std(mean(summary_data.analysis(3).da.c(stable_trial_range,:,:),1),[],3)./ sqrt(size(summary_data.analysis(3).da.c,3)); % 

summary_data.analysis(3).DA_resp.u_avg = mean(mean(summary_data.analysis(3).da.u(stable_trial_range,:,:),1),3);
summary_data.analysis(3).DA_resp.u_sem = std(mean(summary_data.analysis(3).da.u(stable_trial_range,:,:),1),[],3)./ sqrt(size(summary_data.analysis(3).da.c,3)); % ./ sqrt(size(summary_data.analysis(3).da.c,3))

summary_data.analysis(3).DA_resp.o_avg = mean(mean(summary_data.analysis(3).da.o(stable_trial_range,:,:),1),3);
summary_data.analysis(3).DA_resp.o_sem = std(mean(summary_data.analysis(3).da.o(stable_trial_range,:,:),1),[],3)./ sqrt(size(summary_data.analysis(3).da.c,3)); % ./ sqrt(size(summary_data.analysis(3).da.c,3))

shadedErrorBar(-1599:1400,summary_data.analysis(3).DA_resp.o_avg,summary_data.analysis(3).DA_resp.c_sem,{'color',[0 0 1]}); hold on;
shadedErrorBar(-1599:1400,summary_data.analysis(3).DA_resp.c_avg,summary_data.analysis(3).DA_resp.c_sem,{'color',[1 0 0]}); hold on;
shadedErrorBar(-1599:1400,summary_data.analysis(3).DA_resp.u_avg,summary_data.analysis(3).DA_resp.c_sem,{'color',[0 0 0]}); hold on;
plot([-1599 1400],[0 0],'k-.'); hold on;
xlabel('Time from reward (ms)'); ylabel('ACTR predicted DA response (au)');
axis([-1600 1400 -5e-3 15e-3]); box off;

figure(33); clf;
% plot average traces over example intervals:
% 1-100, 200-300, 600-800
ranges = [1:21 ; 40:60 ; 140:160];
for egs_da=1:3

    subplot(1,3,egs_da);
    plot([0 0],[-5e-3 15e-3],'b-'); hold on;
    plot([-1500 -1500],[-5e-3 15e-3],'k-'); hold on;
    for jj=1:size(summary_data.analysis(3).da.c,3)
        tmp_da = mean(summary_data.analysis(3).da.c(ranges(egs_da,:),:,jj),1);
        plot(-1599:1400,tmp_da,'color',[0.5 0.5 0.5 0.5]); hold on;
    end
    plot(-1599:1400, mean( mean(summary_data.analysis(3).da.c(ranges(egs_da,:),:,:),1) ,3) ,'color',[0 0 0],'linewidth',2); hold on;
    axis([-1600 1400 -1e-3 15e-3]); box off;
end

%% 4. Compare DA responses on lick+ and lick- trials predictions
% clear summary_data;
[stim_map] = [1 0 0.67 ; 0 1 0.67 ; 0 0.67 1];
lk_kern = TNC_CreateGaussian(500,40,1000,1);
jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
s_scl = 1;
m_scl = 1;
cnt = 1;

for g=1:numel(run)

    if stim_list(g)==0
        
        % Get DA traces
        num_das = size(run(g).pred_da_move,1);
        for ggg=1:num_das
            summary_data.analysis(4).da.c(ggg,:,cnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
        end
            
        % Get Pr(lick) per same trials
        tcnt=1;
        for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

            summary_data.analysis(4).plck(tcnt,cnt) = run(g).output.pass(gg).plck;
            summary_data.analysis(4).outs(tcnt,:,cnt) = run(g).output.pass(gg).chk.o;
            tcnt=tcnt+1;
        end
        
        cnt = cnt+1;
        
    end

end

%% 4.5 Plot figure panels for lick conditioned DA responses

figure(41); clf;
plot_style  = 'pop'
summary_data.analysis(4).da_lickplus = zeros(3000,1);
summary_data.analysis(4).da_lickminus = zeros(3000,1);

range = 80:140;

for gggg=1:size(summary_data.analysis(4).da.c,3)
    
    lickplus = find(summary_data.analysis(4).plck(range,gggg)>0.9);
    lickminus = find(summary_data.analysis(4).plck(range,gggg)<0.2);

    switch plot_style
        
        case 'individ'
            subplot(6,2,gggg);
            plot(-1599:1400,mean(summary_data.analysis(4).da.c(range(lickplus),:,gggg)),'color',[0 0.67 1]); hold on;
            plot(-1599:1400,mean(summary_data.analysis(4).da.c(range(lickminus),:,gggg)),'color',[1 0 0]); hold on;
            title([num2str(gggg) ' ... ' num2str(numel(lickplus)/numel(lickminus))]);
        case 'pop'
            summary_data.analysis(4).da_lickplus = [summary_data.analysis(4).da_lickplus mean(summary_data.analysis(4).da.c(range(lickplus),:,gggg),1)'];
            summary_data.analysis(4).da_lickminus = [summary_data.analysis(4).da_lickminus mean(summary_data.analysis(4).da.c(range(lickminus),:,gggg),1)'];
            if gggg == size(summary_data.analysis(4).da.c,3)
                shadedErrorBar(-1599:1400,mean(summary_data.analysis(4).da_lickplus,2,'omitnan'),std(summary_data.analysis(4).da_lickplus,[],2,'omitnan')./sqrt(size(summary_data.analysis(4).da.c,3)),{'color',[0 0.67 1]}); hold on;
                shadedErrorBar(-1599:1400,mean(summary_data.analysis(4).da_lickminus,2,'omitnan'),std(summary_data.analysis(4).da_lickminus,[],2,'omitnan')./sqrt(size(summary_data.analysis(4).da.c,3)),{'color',[1 0 0]}); hold on;
            end
    end
    box off; xlabel('Time from reward (ms)'); ylabel('Simulated DA resp. (au)');
    
end

%% 5. PE|lick+ and PE|lick- vs training trials

% Need to analyze these variables:
% net_run.pass(pass).pe   = R_curr(curr_cond)-R_bar(curr_cond);
% net_run.pass(pass).plck = numel(find(anticip_lck>1)) / 10;
% net_run.pass(pass).chk(curr_cond).npi = outputs + curr_input(2,:)*net_out.wIn(net.oUind,2)  + curr_input(1,:)*net_out.wIn(net.oUind,1);

pe_map = TNC_CreateRBColormap(1024,'gp');

scnt = 1;
for g=1:numel(run)

    if stim_list(g)==0 % Only use Cntrl simulations

        cnt = 1;
        for gg=[1:numel(run(g).output.pass)] % Just examine the probed trials

            summary_data.analysis(5).pe(scnt,cnt) = run(g).output.pass(gg).pe;
            summary_data.analysis(5).tcnt(scnt,cnt) = gg;
            summary_data.analysis(5).peI(scnt,cnt) = run(g).output.pass(gg).peI;
            summary_data.analysis(5).plck(scnt,cnt) = run(g).output.pass(gg).plck;
            summary_data.analysis(5).chk(1).npi(scnt,cnt) = run(g).output.pass(gg).chk(1).npi(1605);
            cnt = cnt+1;

        end
        
        scnt = scnt+1;
    end
    
end

figure(50); clf;
subplot(6,1,1);
imagesc(summary_data.analysis(5).pe,[-250 250]); colormap(pe_map);
subplot(6,1,2);
plot(1:size(summary_data.analysis(5).pe,2),mean(summary_data.analysis(5).pe)); axis([0 cnt -200 200]); box off;
subplot(6,1,3);
imagesc(summary_data.analysis(5).plck,[-1 1]); colormap(pe_map);
subplot(6,1,4);
plot(1:size(summary_data.analysis(5).plck,2),sgolayfilt(mean(summary_data.analysis(5).plck),3,11)); box off; axis([0 cnt 0.25 0.75]); 
subplot(6,1,5);
imagesc(summary_data.analysis(5).chk(1).npi,[-10 10]); colormap(pe_map);
subplot(6,1,6);
shadedErrorBar(1:size(summary_data.analysis(5).plck,2),mean(summary_data.analysis(5).chk(1).npi),std(summary_data.analysis(5).chk(1).npi)./sqrt(size(summary_data.analysis(5).chk(1).npi,1))); box off; axis([0 cnt 0 11]); 

figure(51); clf;
summary_data.analysis(5).all_cntrl_plck = summary_data.analysis(5).plck(1:numel(summary_data.analysis(5).plck));
summary_data.analysis(5).all_cntrl_pe = summary_data.analysis(5).pe(1:numel(summary_data.analysis(5).plck));
summary_data.analysis(5).all_cntrl_peI = summary_data.analysis(5).peI(1:numel(summary_data.analysis(5).plck));
summary_data.analysis(5).all_cntrl_tcnt = summary_data.analysis(5).tcnt(1:numel(summary_data.analysis(5).plck));

cnt = 1;
for pp=0:1 %unique(summary_data.analysis(5).all_cntrl_plck)
    summary_data.analysis(5).bin_cntrl_LKperPE.avg(cnt) = median(summary_data.analysis(5).all_cntrl_pe(summary_data.analysis(5).all_cntrl_plck==pp));
    summary_data.analysis(5).bin_cntrl_LKperPE.std(cnt) = std(summary_data.analysis(5).all_cntrl_pe(summary_data.analysis(5).all_cntrl_plck==pp))./sqrt(sum(summary_data.analysis(5).all_cntrl_plck==pp));
    summary_data.analysis(5).bin_cntrl_LKperPE.avgI(cnt) = median(summary_data.analysis(5).all_cntrl_peI(summary_data.analysis(5).all_cntrl_plck==pp));
    summary_data.analysis(5).bin_cntrl_LKperPE.stdI(cnt) = std(summary_data.analysis(5).all_cntrl_peI(summary_data.analysis(5).all_cntrl_plck==pp))./sqrt(sum(summary_data.analysis(5).all_cntrl_plck==pp));
    cnt = cnt+1;
end

summary_data.analysis(5).bin_cntrl_LKperPE.dist = [summary_data.analysis(5).all_cntrl_pe(summary_data.analysis(5).all_cntrl_plck==0)' ; summary_data.analysis(5).all_cntrl_pe(summary_data.analysis(5).all_cntrl_plck==1)'];
summary_data.analysis(5).bin_cntrl_LKperPE.g = [zeros(sum(summary_data.analysis(5).all_cntrl_plck==0),1) ; ones(sum(summary_data.analysis(5).all_cntrl_plck==1),1)];
summary_data.analysis(5).bin_cntrl_LKperPE.distI = [summary_data.analysis(5).all_cntrl_peI(summary_data.analysis(5).all_cntrl_plck==0)' ; summary_data.analysis(5).all_cntrl_peI(summary_data.analysis(5).all_cntrl_plck==1)'];
summary_data.analysis(5).bin_cntrl_LKperPE.tcnt = [summary_data.analysis(5).all_cntrl_tcnt(summary_data.analysis(5).all_cntrl_plck==0)' ; summary_data.analysis(5).all_cntrl_tcnt(summary_data.analysis(5).all_cntrl_plck==1)'];

% plot([0 3],[0 0],'k--'); hold on;
% boxplot(summary_data.analysis(5).bin_cntrl_LKperPE.dist,summary_data.analysis(5).bin_cntrl_LKperPE.g,'labels',{'Lick-' 'Lick+'},'plotstyle','compact'); 
% plot([1.2 1.8],summary_data.analysis(5).bin_cntrl_LKperPE.avg,'k-','linewidth',4); box off;
subplot(121);
plot([-0.5 1.5],[0 0],'k--'); hold on;
swarmchart(summary_data.analysis(5).bin_cntrl_LKperPE.g(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt>400),summary_data.analysis(5).bin_cntrl_LKperPE.dist(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt>400),5,[0.5 0.5 0.5],'MarkerEdgeAlpha',0.2);
plot([0 1],summary_data.analysis(5).bin_cntrl_LKperPE.avg,'ko-','linewidth',4); box off;
ylabel('Mean PE'); axis([-0.5 1.5 -2 2]);
subplot(122);
plot([-0.5 1.5],[0 0],'k--'); hold on;
swarmchart(summary_data.analysis(5).bin_cntrl_LKperPE.g(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt>400),summary_data.analysis(5).bin_cntrl_LKperPE.distI(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt>400),5,[0.5 0.5 0.5],'MarkerEdgeAlpha',0.2);
plot([0 1],summary_data.analysis(5).bin_cntrl_LKperPE.avgI,'ko-','linewidth',4); box off;
ylabel('Mean PEI'); axis([-0.5 1.5 -1 2]);

figure(52); clf; subplot(121);
plot([-2 3],[0 0],'k-'); hold on;
plot([0 0],[-1 3],'k-');
scatter(summary_data.analysis(5).bin_cntrl_LKperPE.dist,summary_data.analysis(5).bin_cntrl_LKperPE.distI,10,'filled','MarkerFaceAlpha',0.25);
ylabel('PE internal'); xlabel('PE veridical'); box off;
subplot(122);
plot([0 800],[0 0],'k-'); hold on;
scatter(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt,summary_data.analysis(5).bin_cntrl_LKperPE.distI,10,summary_data.analysis(5).bin_cntrl_LKperPE.g,'filled','MarkerFaceAlpha',0.25); colormap([pe_map(end-1,:);pe_map(2,:)]);
ylabel('PE'); xlabel('trials'); box off;

[bPE_lminus] = TNC_BinAndMean(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt(summary_data.analysis(5).bin_cntrl_LKperPE.g==0),summary_data.analysis(5).bin_cntrl_LKperPE.distI(summary_data.analysis(5).bin_cntrl_LKperPE.g==0)<0,8);
[bPE_lplus] = TNC_BinAndMean(summary_data.analysis(5).bin_cntrl_LKperPE.tcnt(summary_data.analysis(5).bin_cntrl_LKperPE.g==1),summary_data.analysis(5).bin_cntrl_LKperPE.distI(summary_data.analysis(5).bin_cntrl_LKperPE.g==1)<0,8);

figure(53); clf;
plot([0 800],[0.5 0.5],'k--'); hold on;
errorbar(bPE_lminus.bins.center,bPE_lminus.bins.avg,bPE_lminus.bins.sem,'linewidth',3,'color',pe_map(end-1,:)); hold on;
errorbar(bPE_lplus.bins.center,bPE_lplus.bins.avg,bPE_lplus.bins.sem,'linewidth',3,'color',pe_map(2,:));
axis([0 800 -0.1 1.1]); ylabel('Pr(PE<0)'); xlabel('Trials'); box off;

%% 6. Simulations for stim experiments

global pt_on;
pt_on = 1;

% TUNING UP THE CUE AND REWARD LEARNING RATES
num_sims    = 72;
stim_list   = [ -1*ones(1,12) zeros(1,12) ones(1,12) 20*ones(1,12) 21*ones(1,12) 22*ones(1,12) ];
inits       = repmat(ii([201 130 98 119 141 195]),1,12);
% inits       = repmat(ii([108 119 124 115 109 104]),1,6);
% tmp = repmat([2 2],18,1);
% wIn_vec = tmp(1:numel(tmp));
% tau_vec = tmp(1:numel(tmp))+1;
sat_vec     = repmat(ones(1,6)+6,1,12);
wIn_vec     = repmat(ones(1,6)+2,1,12);
% tau_vec     = repmat(ones(1,6),1,12);
tau_vec     = repmat(rand(1,6)+0.5 ,1,12);
% sat_vec     = repmat(randperm(6)+4,1,9);

clear run;

parfor g = 1:num_sims

    net_init = gens.dets(inits(g)).net % diverse initial states                
    net_init.wIn(net.oUind,:) = [0 wIn_vec(g)];
    tau_trans = tau_vec(g); % now controls eta_wIn learning rate
    filt_scale = 50; % plant scale factor currently
    trans_sat = sat_vec(g);
    stim = stim_list(g);

    [output,net_out,pred_da_sense,pred_da_move,pred_da_move_u,pred_da_sense_u,pred_da_move_o,pred_da_sense_o] = dlRNN_train_learnDA(net_init,input,input_omit,input_uncued,target,act_func_handle,learn_func_handle,transfer_func_handle,65,tau_trans,stim,filt_scale,trans_sat);

    run(g).output = output;
    run(g).net = net_out;
    run(g).pred_da_sense = pred_da_sense;
    run(g).pred_da_move = pred_da_move;
    run(g).pred_da_sense_u = pred_da_sense_u;
    run(g).pred_da_move_u = pred_da_move_u;
    run(g).pred_da_sense_o = pred_da_sense_o;
    run(g).pred_da_move_o = pred_da_move_o;
    disp(['Completed run: ' num2str(g)]);

end

% save ~/'Dropbox (HHMI)'/run-stim2 run stim_list inits

%% 6.5 DA and Licking predictions for stimLick-, stimLick+, Stim++Lick+

clear summary_data;

% Also need to add in 21 and 22 conditions (critic only predictions)
stim_map = [[57 181 74] ; [102 45 145 ] ; [158 31 99] ]/255;
lk_kern = TNC_CreateGaussian(500,40,1000,1);
jrcamp_tau = 500;
t=1:3000;
kern = [zeros(1,3000) exp(-t/jrcamp_tau)];
kern=kern/trapz(kern);
s_scl = 1;
m_scl = 1;

cue_win = 100:600;
rew_win = 1600:2100;

SMcnt = 1;
CTRLcnt = 1;
SPcnt = 1;
SPPcnt = 1;
CritPcnt = 1;
CritMcnt = 1;

for g=1:numel(run)

    switch stim_list(g)
        
        case -1
            
            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).stimMinus.da(ggg,:,SMcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).stimMinus.c_cue_int(ggg,SMcnt) = trapz( summary_data.analysis(6).stimMinus.da(ggg,cue_win,SMcnt) );
                % quantify rew response
                summary_data.analysis(6).stimMinus.c_rew_int(ggg,SMcnt) = trapz( summary_data.analysis(6).stimMinus.da(ggg,rew_win,SMcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).stimMinus.lat(tcnt,SMcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).stimMinus.ant(tcnt,SMcnt) = run(g).output.pass(gg).anticip;
                summary_data.analysis(6).stimMinus.pei(tcnt,SMcnt) = run(g).output.pass(gg).peI;
                summary_data.analysis(6).stimMinus.o(tcnt,SMcnt)    = mean(run(g).output.pass(gg).chk.o(1000:1599));
%                 summary_data.analysis(6).stimMinus.ant(tcnt,SMcnt) = numel(find(run(g).output.pass(gg).chk.v>600 & run(g).output.pass(gg).chk.v<1600));
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).stimMinus.lk(tcnt,:,SMcnt) = conv(tmp_lk,lk_kern,'same');
                
                tcnt=tcnt+1;
                
            end
            SMcnt = SMcnt+1;

        case 0

            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).cntrl.da(ggg,:,CTRLcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).cntrl.c_cue_int(ggg,CTRLcnt) = trapz( summary_data.analysis(6).cntrl.da(ggg,cue_win,CTRLcnt) );
                % quantify rew response
                summary_data.analysis(6).cntrl.c_rew_int(ggg,CTRLcnt) = trapz( summary_data.analysis(6).cntrl.da(ggg,rew_win,CTRLcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).cntrl.lat(tcnt,CTRLcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).cntrl.ant(tcnt,CTRLcnt) = run(g).output.pass(gg).anticip;
                summary_data.analysis(6).cntrl.pei(tcnt,CTRLcnt) = run(g).output.pass(gg).peI;
                summary_data.analysis(6).cntrl.o(tcnt,CTRLcnt) = mean(run(g).output.pass(gg).chk.o(1000:1599));
                summary_data.analysis(6).cntrl.antC(tcnt,CTRLcnt) = numel(find(run(g).output.pass(gg).chk.v>600 & run(g).output.pass(gg).chk.v<1600));
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).cntrl.lk(tcnt,:,CTRLcnt) = conv(tmp_lk,lk_kern,'same');
                summary_data.analysis(6).cntrl.antC(tcnt,CTRLcnt) = max( summary_data.analysis(6).cntrl.lk(tcnt,100:600,CTRLcnt) );
                
                tcnt=tcnt+1;
                
            end
            CTRLcnt = CTRLcnt+1;
            
        case 1

            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).stimPlus.da(ggg,:,SPcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).stimPlus.c_cue_int(ggg,SPcnt) = trapz( summary_data.analysis(6).stimPlus.da(ggg,cue_win,SPcnt) );
                % quantify rew response
                summary_data.analysis(6).stimPlus.c_rew_int(ggg,SPcnt) = trapz( summary_data.analysis(6).stimPlus.da(ggg,rew_win,SPcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).stimPlus.lat(tcnt,SPcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).stimPlus.ant(tcnt,SPcnt) = run(g).output.pass(gg).anticip;
                summary_data.analysis(6).stimPlus.pei(tcnt,SPcnt) = run(g).output.pass(gg).peI;
                summary_data.analysis(6).stimPlus.o(tcnt,SPcnt) = mean(run(g).output.pass(gg).chk.o(1000:1599));
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).stimPlus.lk(tcnt,:,SPcnt) = conv(tmp_lk,lk_kern,'same');
                
                tcnt=tcnt+1;
                
            end
            SPcnt = SPcnt+1;
            
        case 20
            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).stimPlusPlus.da(ggg,:,SPPcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).stimPlusPlus.c_cue_int(ggg,SPPcnt) = trapz( summary_data.analysis(6).stimPlusPlus.da(ggg,cue_win,SPPcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).stimPlusPlus.lat(tcnt,SPPcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).stimPlusPlus.ant(tcnt,SPPcnt) = run(g).output.pass(gg).anticip;
                summary_data.analysis(6).stimPlusPlus.o(tcnt,SPPcnt) = mean(run(g).output.pass(gg).chk.o(1000:1599));
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).stimPlusPlus.lk(tcnt,:,SPPcnt) = conv(tmp_lk,lk_kern,'same');
                summary_data.analysis(6).stimPlusPlus.antC(tcnt,SPPcnt) = max( summary_data.analysis(6).stimPlusPlus.lk(tcnt,100:600,SPPcnt) );
                
                tcnt=tcnt+1;
                
            end
            SPPcnt = SPPcnt+1;

        case 21
                        
            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).critP.da(ggg,:,CritPcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).critP.c_cue_int(ggg,CritPcnt) = trapz( summary_data.analysis(6).critP.da(ggg,cue_win,CritPcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).critP.lat(tcnt,CritPcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).critP.ant(tcnt,CritPcnt) = run(g).output.pass(gg).anticip;
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).critP.lk(tcnt,:,CritPcnt) = conv(tmp_lk,lk_kern,'same');
                
                tcnt=tcnt+1;
                
            end
            CritPcnt = CritPcnt+1;

        case 22

            num_das = size(run(g).pred_da_move,1);
            for ggg=1:num_das
                
                % continuous plot
                summary_data.analysis(6).critM.da(ggg,:,CritMcnt) = conv( s_scl*run(g).pred_da_sense(ggg,:) + m_scl*run(g).pred_da_move(ggg,:) , kern , 'same');
                
                % quantify cue response
                summary_data.analysis(6).critM.c_cue_int(ggg,CritMcnt) = trapz( summary_data.analysis(6).critM.da(ggg,cue_win,CritMcnt) );

            end
            
            tcnt=1;
            for gg=[1 run(g).net.update:run(g).net.update:numel(run(g).output.pass)] % Just examine the probed trials

                summary_data.analysis(6).critM.lat(tcnt,CritMcnt) = run(g).output.pass(gg).lat;
                summary_data.analysis(6).critM.ant(tcnt,CritMcnt) = run(g).output.pass(gg).anticip;
                
                tmp_lk = zeros(1,3000);
                tmp_lk(unique(run(g).output.pass(gg).chk.v)) = 1;
                summary_data.analysis(6).critM.lk(tcnt,:,CritMcnt) = conv(tmp_lk,lk_kern,'same');
                
                tcnt=tcnt+1;
                
            end
            CritMcnt = CritMcnt+1;
            
    end


end

% 6.75 Plot the various figure panels for stimLick experiments

% Model overview
    figure(600); clf;

    subplot(221);
    plot([1 5:5:800],mean(summary_data.analysis(6).cntrl.ant,2),'k');
    hold on;    
    plot([1 5:5:800],mean(summary_data.analysis(6).stimMinus.ant,2),'color',stim_map(1,:));
    plot([1 5:5:800],mean(summary_data.analysis(6).stimPlus.ant,2),'-','color',stim_map(2,:));
%     plot([1 5:5:800],mean(summary_data.analysis(6).stimPlusPlus.ant,2),'--','color',stim_map(3,:));
ylabel('Anticipatory licking')

    subplot(222);
    plot([1 5:5:800],mean(summary_data.analysis(6).cntrl.c_cue_int,2),'k');
    hold on;    
    plot([1 5:5:800],mean(summary_data.analysis(6).stimMinus.c_cue_int,2),'color',stim_map(1,:));
    plot([1 5:5:800],mean(summary_data.analysis(6).stimPlus.c_cue_int,2),'-','color',stim_map(2,:));
ylabel('Cued DA response')

    subplot(223);
    plot([1 5:5:800],mean(summary_data.analysis(6).cntrl.pei,2),'k');
    hold on;    
    plot([1 5:5:800],mean(summary_data.analysis(6).stimMinus.pei,2),'color',stim_map(1,:));
    plot([1 5:5:800],mean(summary_data.analysis(6).stimPlus.pei,2),'-','color',stim_map(2,:));
ylabel('Error_r')

subplot(224);
    plot([1 5:5:800],mean(summary_data.analysis(6).cntrl.o,2),'k-','linewidth',2);
    hold on;    
    plot([1 5:5:800],mean(summary_data.analysis(6).stimMinus.o,2),'-','color',stim_map(1,:),'linewidth',2);
    plot([1 5:5:800],mean(summary_data.analysis(6).stimPlus.o,2),'-','color',stim_map(2,:),'linewidth',2);
    plot([1 5:5:800],mean(summary_data.analysis(6).stimPlusPlus.o,2),'-','color',stim_map(3,:),'linewidth',2);
ylabel('Policy output')

% Figure panels for manuscript:
figure(601); clf; % cued DA - cntrl & cued lick - cntrl, 100 trials bins for StimLick+ / StimLick-
cntrs = 11:20:151;
for qq=1:8 % hundred trial bins
    
    curr_bin = [-9:10] + cntrs(qq);

    summary_data.analysis(6).lk.cntrl_avg(qq) = mean( mean(summary_data.analysis(6).cntrl.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.cntrlC_avg(qq) = mean( mean(summary_data.analysis(6).cntrl.antC(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.cntrl_sd(qq) = std( mean(summary_data.analysis(6).cntrl.ant(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).lk.cntrlC_sd(qq) = std( mean(summary_data.analysis(6).cntrl.antC(curr_bin,:),1) ,[],2);

    summary_data.analysis(6).lk.minus_avg(qq) = mean( mean(summary_data.analysis(6).stimMinus.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.minus_sd(qq) = std( mean(summary_data.analysis(6).stimMinus.ant(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).lk.plus_avg(qq) = mean( mean(summary_data.analysis(6).stimPlus.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.plus_sd(qq) = std( mean(summary_data.analysis(6).stimPlus.ant(curr_bin,:),1) ,[],2);

    summary_data.analysis(6).lk.pplus_avgS(qq) = mean( mean(summary_data.analysis(6).stimPlusPlus.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.pplus_sdS(qq) = std( mean(summary_data.analysis(6).stimPlusPlus.ant(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).lk.pplus_avg(qq) = mean( mean(summary_data.analysis(6).stimPlusPlus.antC(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.pplus_sd(qq) = std( mean(summary_data.analysis(6).stimPlusPlus.antC(curr_bin,:),1) ,[],2);

    
    summary_data.analysis(6).lat.cntrl_avg(qq) = mean( mean(summary_data.analysis(6).cntrl.lat(curr_bin,:),1) ,2);
    summary_data.analysis(6).lat.pplus_avg(qq) = mean( mean(summary_data.analysis(6).stimPlusPlus.lat(curr_bin,:),1) ,2);
    
    
    summary_data.analysis(6).lk.cminus_avg(qq) = mean( mean(summary_data.analysis(6).critM.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.cminus_sd(qq) = std( mean(summary_data.analysis(6).critM.ant(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).lk.cplus_avg(qq) = mean( mean(summary_data.analysis(6).critP.ant(curr_bin,:),1) ,2);
    summary_data.analysis(6).lk.cplus_sd(qq) = std( mean(summary_data.analysis(6).critP.ant(curr_bin,:),1) ,[],2);
    
    summary_data.analysis(6).cueDA.cntrl_avg(qq) = mean( mean(summary_data.analysis(6).cntrl.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.cntrl_sd(qq) = std( mean(summary_data.analysis(6).cntrl.c_cue_int(curr_bin,:),1) ,[],2);

    summary_data.analysis(6).cueDA.minus_avg(qq) = mean( mean(summary_data.analysis(6).stimMinus.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.minus_sd(qq) = std( mean(summary_data.analysis(6).stimMinus.c_cue_int(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).cueDA.plus_avg(qq) = mean( mean(summary_data.analysis(6).stimPlus.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.plus_sd(qq) = std( mean(summary_data.analysis(6).stimPlus.c_cue_int(curr_bin,:),1) ,[],2);

    summary_data.analysis(6).cueDA.pplus_avg(qq) = mean( mean(summary_data.analysis(6).stimPlusPlus.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.pplus_sd(qq) = std( mean(summary_data.analysis(6).stimPlusPlus.c_cue_int(curr_bin,:),1) ,[],2);

    summary_data.analysis(6).cueDA.cminus_avg(qq) = mean( mean(summary_data.analysis(6).critM.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.cminus_sd(qq) = std( mean(summary_data.analysis(6).critM.c_cue_int(curr_bin,:),1) ,[],2);
    summary_data.analysis(6).cueDA.cplus_avg(qq) = mean( mean(summary_data.analysis(6).critP.c_cue_int(curr_bin,:),1) ,2);
    summary_data.analysis(6).cueDA.cplus_sd(qq) = std( mean(summary_data.analysis(6).critP.c_cue_int(curr_bin,:),1) ,[],2);


end

summary_data.analysis(6).lk.minusD_avg = summary_data.analysis(6).lk.minus_avg - summary_data.analysis(6).lk.cntrl_avg;
summary_data.analysis(6).lk.plusD_avg = summary_data.analysis(6).lk.plus_avg - summary_data.analysis(6).lk.cntrl_avg;

summary_data.analysis(6).cueDA.minusD_avg = summary_data.analysis(6).cueDA.minus_avg - summary_data.analysis(6).cueDA.cntrl_avg;
summary_data.analysis(6).cueDA.plusD_avg = summary_data.analysis(6).cueDA.plus_avg - summary_data.analysis(6).cueDA.cntrl_avg;

summary_data.analysis(6).lk.cminusD_avg = summary_data.analysis(6).lk.cminus_avg - summary_data.analysis(6).lk.cntrl_avg;
summary_data.analysis(6).lk.cplusD_avg = summary_data.analysis(6).lk.cplus_avg - summary_data.analysis(6).lk.cntrl_avg;

summary_data.analysis(6).cueDA.cminusD_avg = summary_data.analysis(6).cueDA.cminus_avg - summary_data.analysis(6).cueDA.cntrl_avg;
summary_data.analysis(6).cueDA.cplusD_avg = summary_data.analysis(6).cueDA.cplus_avg - summary_data.analysis(6).cueDA.cntrl_avg;

figure(601); clf; % cued DA - cntrl & cued lick - cntrl, 100 trials bins for StimLick+ / StimLick-
subplot(211);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.minusD_avg,summary_data.analysis(6).cueDA.minus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(1,:));
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.plusD_avg,summary_data.analysis(6).cueDA.plus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(2,:));
axis([0 800 -0.3 0.3]); box off; xlabel('Trials'); ylabel('Cued DA resp. - cntrl (au)');

subplot(212);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.minusD_avg,summary_data.analysis(6).lk.minus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(1,:));
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.plusD_avg,summary_data.analysis(6).lk.plus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(2,:));
axis([0 800 -2 2]); box off; xlabel('Trials'); ylabel('Cued licks - cntrl (Hz)');

figure(602); clf; % cued DA - cntrl & cued lick - cntrl, 100 trials bins for StimLick+ / StimLick- if DA==PE (case 21 and 22)
subplot(211);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.cminusD_avg,summary_data.analysis(6).cueDA.cminus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(1,:));
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.cplusD_avg,summary_data.analysis(6).cueDA.cplus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(2,:));
axis([0 800 -0.3 0.3]); box off; xlabel('Trials'); ylabel('Cued DA resp. - cntrl (au)');

subplot(212);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.cminusD_avg,summary_data.analysis(6).lk.cminus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(1,:));
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.cplusD_avg,summary_data.analysis(6).lk.cplus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(2,:));
axis([0 800 -2 2]); box off; xlabel('Trials'); ylabel('Cued licks - cntrl (Hz)');

figure(603); clf; % cued DA & cued licking for stim and cntrl over learning, 100 trials bins for Stim++Lick+ (case 20)
subplot(211);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.cntrl_avg,summary_data.analysis(6).cueDA.cntrl_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color','k');
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.pplus_avg,summary_data.analysis(6).cueDA.pplus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(3,:));
axis([0 800 -0.1 0.5]); box off; xlabel('Trials'); ylabel('Cued DA resp. (au)');

subplot(212);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.cntrlC_avg,summary_data.analysis(6).lk.cntrlC_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color','k');
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.pplus_avg,summary_data.analysis(6).lk.pplus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(3,:));
% axis([0 800 1 7]); 
box off; xlabel('Trials'); ylabel('Cued licks (Hz)');


figure(604); clf;

subplot(2,2,1);
plot(mean(mean(summary_data.analysis(6).stimPlusPlus.da(20:60,:,:),1),3),'linewidth',2,'color',stim_map(3,:)); hold on;
plot(mean(mean(summary_data.analysis(6).cntrl.da(20:60,:,:),1),3),'linewidth',2,'color','k'); hold on;
axis([0 1000 -1e-4 1.5e-3]); box off;

subplot(2,2,2);
plot(mean(mean(summary_data.analysis(6).stimPlusPlus.da(80:120,:,:),1),3),'linewidth',2,'color',stim_map(3,:)); hold on;
plot(mean(mean(summary_data.analysis(6).cntrl.da(80:120,:,:),1),3),'linewidth',2,'color','k'); hold on;
axis([0 1000 -1e-4 1.5e-3]); box off;

subplot(2,2,3);
plot(mean(mean(summary_data.analysis(6).stimPlusPlus.lk(20:60,:,:),1),3),'linewidth',2,'color',stim_map(3,:)); hold on;
plot(mean(mean(summary_data.analysis(6).cntrl.lk(20:60,:,:),1),3),'linewidth',2,'color','k'); hold on;
axis([0 2000 -1e-4 9e-3]); box off;

subplot(2,2,4);
plot(mean(mean(summary_data.analysis(6).stimPlusPlus.lk(80:120,:,:),1),3),'linewidth',2,'color',stim_map(3,:)); hold on;
plot(mean(mean(summary_data.analysis(6).cntrl.lk(80:120,:,:),1),3),'linewidth',2,'color','k'); hold on;
axis([0 2000 -1e-4 9e-3]); box off;


% SUPPLEMENT FIGURE
figure(605); clf; % cued DA & cued licking for stim and cntrl over learning, 100 trials bins for Stim++Lick+ (case 20)
subplot(311);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.cntrl_avg,summary_data.analysis(6).cueDA.cntrl_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color','k');
errorbar((cntrs-1)*5,summary_data.analysis(6).cueDA.pplus_avg,summary_data.analysis(6).cueDA.pplus_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(3,:));
axis([0 800 -0.1 0.5]); box off; xlabel('Trials'); ylabel('Cued DA resp. (au)');

subplot(312);
plot([0 800],[0 0],'k--'); hold on;
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.cntrl_avg,summary_data.analysis(6).lk.cntrl_sd./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color','k');
errorbar((cntrs-1)*5,summary_data.analysis(6).lk.pplus_avgS,summary_data.analysis(6).lk.pplus_sdS./sqrt(size(summary_data.analysis(6).cntrl.ant,2)),'linewidth',2,'color',stim_map(3,:));
% axis([0 800 1 7]); 
box off; xlabel('Trials'); ylabel('Cued licks (Hz)');

subplot(313);
plot([0 800],[0 0],'k--'); hold on;
plot([0 (cntrs-1)*5],[mean(summary_data.analysis(6).cntrl.lat(1,:)) summary_data.analysis(6).lat.cntrl_avg],'linewidth',2,'color','k');
plot([0 (cntrs-1)*5],[mean(summary_data.analysis(6).stimPlusPlus.lat(1,:)) summary_data.analysis(6).lat.pplus_avg],'linewidth',2,'color',stim_map(3,:));
box off; xlabel('Trials'); ylabel('Collect Latency (ms)');
    

%% 6.5 Plot figure panels for comparisons

inds2use = 1:18;

for qq=1:9 % hundred trial bins
    
    if qq==1
        curr_bin = 1:2;
    else
        curr_bin=[1:20] + (qq-2)*20;
    end
    this_bin_avg_sm = mean(summary_data.analysis(6).stimMinus.ant(curr_bin,inds2use),1);
    this_bin_avg_sp = mean(summary_data.analysis(6).stimPlus.ant(curr_bin,inds2use),1);
    this_bin_avg_spp = mean(summary_data.analysis(6).stimPlusPlus.ant(curr_bin,inds2use),1);
    this_bin_avg_ctrl = mean(summary_data.analysis(6).cntrl.ant(curr_bin,inds2use),1);

    summary_data.analysis(6).ant_sm.avg(qq) = mean(this_bin_avg_sm-this_bin_avg_ctrl);
    summary_data.analysis(6).ant_sm.sem(qq) = std(this_bin_avg_sm-this_bin_avg_ctrl)./sqrt(numel(this_bin_avg_ctrl));
    summary_data.analysis(6).ant_sp.avg(qq) = mean(this_bin_avg_sp-this_bin_avg_ctrl);
    summary_data.analysis(6).ant_sp.sem(qq) = std(this_bin_avg_sp-this_bin_avg_ctrl)./sqrt(numel(this_bin_avg_ctrl));

    summary_data.analysis(6).ant_ctrl.avg(qq) = mean(this_bin_avg_ctrl);
    summary_data.analysis(6).ant_ctrl.sem(qq) = std(this_bin_avg_ctrl)./sqrt(numel(this_bin_avg_ctrl));
    summary_data.analysis(6).ant_spp.avg(qq) = mean(this_bin_avg_spp);
    summary_data.analysis(6).ant_spp.sem(qq) = std(this_bin_avg_spp)./sqrt(numel(this_bin_avg_spp));
    
    this_bin_da_sm = mean(summary_data.analysis(6).stimMinus.c_cue_int(curr_bin,inds2use),1);
    this_bin_da_sp = mean(summary_data.analysis(6).stimPlus.c_cue_int(curr_bin,inds2use),1);
    this_bin_da_spp = mean(summary_data.analysis(6).stimPlusPlus.c_cue_int(curr_bin,inds2use),1);
    this_bin_da_ctrl = mean(summary_data.analysis(6).cntrl.c_cue_int(curr_bin,inds2use),1);

    summary_data.analysis(6).da_sm.avg(qq) = mean(this_bin_da_sm-this_bin_da_ctrl);
    summary_data.analysis(6).da_sm.sem(qq) = std(this_bin_da_sm-this_bin_da_ctrl)./sqrt(numel(this_bin_da_ctrl));
    summary_data.analysis(6).da_sp.avg(qq) = mean(this_bin_da_sp-this_bin_da_ctrl);
    summary_data.analysis(6).da_sp.sem(qq) = std(this_bin_da_sp-this_bin_da_ctrl)./sqrt(numel(this_bin_da_ctrl));

    summary_data.analysis(6).da_ctrl.avg(qq) = mean(this_bin_da_ctrl);
    summary_data.analysis(6).da_ctrl.sem(qq) = std(this_bin_da_ctrl)./sqrt(numel(this_bin_da_ctrl));
    summary_data.analysis(6).da_spp.avg(qq) = mean(this_bin_da_spp);
    summary_data.analysis(6).da_spp.sem(qq) = std(this_bin_da_spp)./sqrt(numel(this_bin_da_spp));
    
end

figure(61); clf;
subplot(212);
plot([0 800],[0 0],'k--'); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).ant_sm.avg,summary_data.analysis(6).ant_sm.sem,{'color',stim_map(1,:)}); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).ant_sp.avg,summary_data.analysis(6).ant_sp.sem,{'color',stim_map(2,:)}); hold on;
axis([0 800 -4 4]);
ylabel('cued licks - ctrl (Hz)');
xlabel('trials'); box off;

subplot(211);
plot([0 800],[0 0],'k--'); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).da_sm.avg,summary_data.analysis(6).da_sm.sem,{'color',stim_map(1,:)}); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).da_sp.avg,summary_data.analysis(6).da_sp.sem,{'color',stim_map(2,:)}); hold on;
axis([0 800 -1 1]);
ylabel('cued DA - ctrl (au)');
xlabel('trials'); box off;

figure(62); clf;
subplot(212);
plot([0 800],[0 0],'k--'); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).ant_ctrl.avg,summary_data.analysis(6).ant_ctrl.sem,{'color',[0 0 0]}); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).ant_spp.avg,summary_data.analysis(6).ant_spp.sem,{'color',stim_map(3,:)}); hold on;
axis([0 800 0 6]);
ylabel('cued licks - ctrl (Hz)');
xlabel('trials'); box off;

subplot(211);
plot([0 800],[0 0],'k--'); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).da_ctrl.avg,summary_data.analysis(6).da_ctrl.sem,{'color',[0 0 0]}); hold on;
shadedErrorBar(0:100:800,summary_data.analysis(6).da_spp.avg,summary_data.analysis(6).da_spp.sem,{'color',stim_map(3,:)}); hold on;
axis([0 800 -1 1]);
ylabel('cued DA - ctrl (au)');
xlabel('trials'); box off;

%% DEPRECATED CODE
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

for g=[1 13 25 37]
% for g=[1 7]+1

    switch stim_list(g)
        case -1
            figure('Name',['Init ' num2str(inits(g)) ' : StimLick-' ],'NumberTitle','off'); clf;
        case 0
            figure('Name',['Init ' num2str(inits(g)) ' : Cntrl' ],'NumberTitle','off'); clf;
        case 1
            figure('Name',['Init ' num2str(inits(g)) ' : StimLick+' ],'NumberTitle','off'); clf;
        case 20
            figure('Name',['Init ' num2str(inits(g)) ' : Stim++Lick+' ],'NumberTitle','off'); clf;
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
    for gg=[run(g).net.update:run(g).net.update:numel(run(g).output.pass)]
        summary_data.lat(g).la(cnt,:) = [run(g).output.pass(gg).lat run(g).output.pass(gg).lat_u run(g).output.pass(gg).lat_o];
        summary_data.lik(g).lk(cnt,unique(run(g).output.pass(gg).chk.v)) = 1;
        summary_data.lik(g).lks(cnt,:) = conv(summary_data.lik(g).lk(cnt,:),lk_kern,'same');

        summary_data.pe(g).pe(cnt) = run(g).output.pass(gg).pe;
        summary_data.plck(g).plck(cnt) = run(g).output.pass(gg).plck;

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
    axis([0 3000 -1 20]);
    ylabel('DA resp. (a.u.)');  xlabel('Time'); legend({'1-300' '300-600'},'location','northwest');

end
%
figure(31); clf;
subplot(2,4,[1:3 5:7]);
range = 20:20:140;
tc_sl = trapz(summary_data.analysis(2).da_sl.c,2);
plot([0 max(range)*5],[0 0],'k--'); hold on;
plot(range*5,mean(tc_sl(range,:,:),3),'o-','linewidth',2,'color',[0 0.67 1],'MarkerFaceColor','w'); hold on;
tc = trapz(summary_data.analysis(2).da.c,2);
plot(range*5,mean(tc(range,:,:),3),'ko-','linewidth',2,'MarkerFaceColor','w'); hold on;
% shadedErrorBar(range*5,mean(tc(range,:,:),3),std(tc(range,:,:),[],3)); hold on;
xlabel('Trials'); ylabel('Integrated cue response ACTR simulation (a.u.)');
box off;
axis([90 10+max(range)*5 -1 6]);

mean_range = 120:121;
tmp_sl = mean(summary_data.analysis(2).da_sl.c(mean_range,:,:),3);
tmp = mean(   summary_data.analysis(2).da.c(mean_range,:,:),3);
subplot(2,4,4);
shadedErrorBar(-1600:1399,mean(tmp_sl,1),std(tmp_sl,[],1)); hold on; axis([-1700 1500 -0.001 0.011]);
box off;

subplot(2,4,8);
shadedErrorBar(-1600:1399,mean(tmp,1),std(tmp,[],1)); hold on; axis([-1700 1500 -0.001 0.011]);
box off; xlabel('Time from stim (ms)'); ylabel('ACTR simulated DA response (a.u.)');

% Summary display plot for talks of training experience.

% To plot:

% Trialwise / sessionwise:
% Anticipatory licking
% Fraction lick+ trials
%     per trial lick analysis is here:
%     run(1).output.pass(1).chk.v
% Performance errors on lick+ / lick- trials (maybe seeing the sign effect
% that stimLick exploits?)

% cued vs uncued latency
%     run(1).output.pass(1).lat
%     run(1).output.pass(1).lat_u
%     run(1).output.pass(1).lat_o

% Comparison across distinct run types
stim_cat = unique(stim_list);
stim_labels = { 'Lick-','Control','Lick+','Stim++Lick+' };
figure(601); clf;
            
    for sg = 1:numel(stim_cat)
        inds = find(stim_list==stim_cat(sg));
        plot(0:5:800,sgolayfilt(mean(model_runs.anticip(inds,:)),3,7),'color',stim_map(sg,:),'linewidth',2); hold on;
    end
    legend(stim_labels(1:numel(stim_cat)));

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
