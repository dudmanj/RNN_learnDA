%% Example analysis code from Figure 1 of Coddington, Lindo, Dudman

load('~/Dropbox (HHMI)/Conditioned responding and DA/seshMerge.mat')

control=[1 2 4 6 9 11 15 16 19];
stimPlus=[7 8 12 13 17];
stimMinus=[3 5 10 14 18 20];
final_lat=[16 2 11 1 4 15 6 9 19];


goodPerf=control([1 2 6 8])
badPer = control([4 5 7 9])

exampFig1 = control([2 4 8]);
eF1_map = [113 120 139 ; 206 26 43 ; 201 180 125];

%% compute a trialUse field to decide what trials have valid data (no NaN).
for q=1:19 %numel(seshMerge)
    tmp(q) = numel(seshMerge(q).trialID);
    seshMerge(q).trialUse = ones(size(seshMerge(q).trialID));
    seshMerge(q).trialUse(isnan(seshMerge(q).nose(:,1))) = 0;
end
max(tmp)

%% New version
beta_vec_type = 'pca'
trial_filt = [0 ones(1,5)/5 0];
clear *_dat allMice_L*
sust_dat_all = [];
trans_dat_all = [];
figure(1); clf; 
figure(2); clf;
figure(3); clf;
figure(4); clf;
cost_on = 1;
mID_cmap = TNC_CreateRBColormap(9,'cpb');

for k=control
    
    sust_dat_all = [ sust_dat_all ; double( seshMerge(k).predVars(:,1:3) ) ];
    trans_dat_all = [ trans_dat_all ; double( seshMerge(k).predVars(:,5:7) ) ];
    
end

switch beta_vec_type
    
    case 'pca'    
        [trans_pc.V, trans_pc.M] = compute_mapping(trans_dat_all , 'PCA',1);
        [sust_pc.V, sust_pc.M] = compute_mapping(sust_dat_all , 'PCA',1);

    case 'da'
        trans_pc.V = trans_dat*B.beta.trn;
        sust_pc.V = sust_dat*B.beta.sus;
        
end



for k=control
    
    ind = find(control==k);
    lat_ind = find(final_lat==k);    
    if numel(find(k==goodPerf)>0)
        wide = 1.5;
        allMice_Learn.goodbad(ind) = 1;
    else
        wide = 1.5;
        allMice_Learn.goodbad(ind) = 0;
    end
    
    if numel(find(k==exampFig1)>0)
        exFig1col = eF1_map(find(k==exampFig1),:) / 255;
        wide = 5;
    else
        exFig1col = [0.5 0.5 0.5];
        wide = 1;
    end
    
    for kk=1:8
        seshMerge(k).predVarsS(:,kk) = conv(seshMerge(k).predVars(:,kk),[0 ones(1,5) 0]/5,'same');
    end
    
    trans_dat = double( seshMerge(k).predVarsS(:,5:7) );
    sust_dat = double( seshMerge(k).predVarsS(:,1:3) );
    latency = double( seshMerge(k).predVarsS(:,8) );
        latency(latency>3000) = 3000; 
        mean(latency(1:10))
    
    % Project onto 1st PC for transient and sustained
    transient = trans_dat*trans_pc.M.M(:,1);
    sustained = sust_dat*sust_pc.M.M(:,1);
    
    % Compute offset for fitting exponential
    trans_off = median(transient(600:800,1));
    sust_off = median(sustained(600:800,1));
    lat_off = median(latency(600:800,1));
    
    final_latency.data(ind) = lat_off;
    
    trial_rng = [1:800]';
    
    % Fit an exponential to the data
    sustained_model = fit( trial_rng , sustained-sust_off,'exp2','Lower',[-4 -0.1 -4 -0.1],'Upper',[5 0.1 5 0.1]);    
    transient_model = fit( trial_rng , transient-trans_off,'exp2','Lower',[50 -0.05 50 -0.5],'Upper',[500 0 500 0]);
    latency_model = fit( trial_rng , latency-lat_off,'exp2','Upper',[3000 0 3000 0]);
    
    transient_sm = transient_model(trial_rng)+trans_off;
    sustained_sm = sustained_model(trial_rng)+sust_off;
    latency_sm = latency_model(trial_rng)+lat_off;
    
    allMice_Learn.lm(ind,:) = latency_sm;
    allMice_Learn.sm(ind,:) = sustained_sm;
    allMice_Learn.tm(ind,:) = transient_sm;

    allMice_Learn.lmS(ind,:) = sgolayfilt(latency,3,151);
    allMice_Learn.smS(ind,:) = sgolayfilt(sustained,3,151);
    allMice_Learn.tmS(ind,:) = sgolayfilt(transient,3,151);

    allMice_Learn.tm_tau(1,ind) = transient_model.b;
    allMice_Learn.tm_tau(2,ind) = transient_model.a;
    allMice_Learn.sm_tau(1,ind) = sustained_model.b;
    allMice_Learn.sm_tau(2,ind) = sustained_model.a;

    % Examine model fit quality
    figure(10+k); clf; 
        subplot(131); plot(transient); hold on; plot(transient_sm,'linewidth',3); axis([0 300 0 400]);
        subplot(132); plot(sustained); hold on; plot(sustained_sm,'linewidth',3); axis([0 800 0 8]);
        subplot(133); semilogy(latency); hold on; semilogy(latency_sm,'linewidth',3); axis([0 800 70 3000]);

        trans.decay.y(k) = transient_sm(end) + 0.25*(transient_sm(1)-transient_sm(end));
        trans.decay.x(k) = find( transient_sm > transient_sm(end) + 0.25*(transient_sm(1)-transient_sm(end)) , 1 , 'last' );
        sust.decay.y(k) = sustained_sm(1) + 0.75*(sustained_sm(end)-sustained_sm(1));
        if sustained_sm(end)-sustained_sm(1) < 0
            sust.decay.x(k) = 800;
        else
            sust.decay.x(k) = find( sustained_sm > sustained_sm(1) + 0.75*(sustained_sm(end)-sustained_sm(1)) , 1 , 'first' );
        end

        allMice_Learn.tm_75p(ind) = trans.decay.x(k);
        allMice_Learn.sm_75p(ind) = sust.decay.x(k);
        allMice_Learn.tm_init(ind) = transient_sm(1);
        allMice_Learn.rank_final_lat(ind) = find(final_lat==k);
        allMice_Learn.final_lat(ind) = lat_off;

    figure(3);
        set(0,'DefaultFigureRenderer','painters');
        plot(transient_sm,'-','linewidth',wide,'color',[exFig1col,0.75]); hold on;
        plot(trans.decay.x(k),trans.decay.y(k),'ko');
        disp(['Mouse ' num2str(k) ...
            ':  transient time to 75%=' num2str(trans.decay.x(k)) '   ' ...
            ':  sustained time to 75%=' num2str(sust.decay.x(k)) ...
            ]);
        
    figure(4);
        set(0,'DefaultFigureRenderer','painters');
        plot(sustained_sm,'-','linewidth',wide,'color',[exFig1col,0.75]); hold on;
    
    figure(2);
        set(0,'DefaultFigureRenderer','painters');
        switch cost_on
            case 0
                plot3(sustained_sm,transient_sm,(latency_sm),'-','linewidth',wide,'color',[exFig1col]); hold on;
            case 1
                plot3(sustained_sm,transient_sm,1-exp(-latency_sm/500),'-','linewidth',wide,'color',[mID_cmap(lat_ind,:),0.75]); hold on;
        end
        
end

    figure(2);
        set(0,'DefaultFigureRenderer','painters');

box on; view(48,30);
set(gca,'Color',[0.95 0.95 0.95]);
grid on;
ylabel('Reactive');
xlabel('Anticipatory');
switch cost_on
    case 0
        zlabel('Latency (ms)');
        axis([0 4 0 275 2 3000]);
    case 1
        zlabel('Cost');
        axis([-2 4 0 275 0 1]);
end

map_c = TNC_CreateRBColormap(9,'cpb');
p = ranksum(allMice_Learn.tm_75p' , allMice_Learn.sm_75p');
figure(100); clf; subplot(131);
boxplot([allMice_Learn.tm_75p' allMice_Learn.sm_75p'],'labels',{'Reactive', 'Anticipatory'});
ylabel('Trials to 75%'); hold on;
for gg=1:numel(allMice_Learn.tm_75p)
    plot([1.2 1.8],[allMice_Learn.tm_75p(gg) allMice_Learn.sm_75p(gg)],'k-','linewidth',2,'color',map_c(allMice_Learn.rank_final_lat(gg),:));
end
text(1.5,0,num2str(p));
subplot(132);
plot(allMice_Learn.final_lat,allMice_Learn.sm_75p,'ko'); ylabel('Anticip (t75)'); xlabel('Final latency rank'); box off;
subplot(133);
plot(allMice_Learn.final_lat,allMice_Learn.sm_75p-allMice_Learn.tm_75p,'ko'); ylabel('Anticip - React'); xlabel('Final latency rank'); box off;
