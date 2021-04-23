function [checks,varargout] = dlRNN_Pcheck_transfer(activity,filt_scale)

option = 'state_simple';
% trans_prob = 'non-linear';
trans_prob = 'pass-thru';
plotFlag = 0;
reaction_time = 150;

% Original idea was just to scale this down
plant_scale = filt_scale; 

% However a better version is really a high pass filter of activity
filter1 = TNC_CreateGaussian(100,7,200,1);
filter1(1:100) = 0;
filter2 = TNC_CreateGaussian(100,10,200,1)./1.07;
filter2(1:100) = 0;
filter = filter1-filter2;
% figure(200); plot(filter);
% figure(201); hold on; plot(conv([zeros(1,500) ones(1,1000)],filter,'same'));

% transform normalized activity into rate parameter for lick plant
switch trans_prob
    
    case 'pass-thru'
        activity(activity<0) = 0;
        activity = activity ./ plant_scale;
        
    case 'non-linear'
        kn = TNC_CreateGaussian(500,150,1000,1);
        kn_c = cumsum(kn);
        re_act = activity.*1000;
        re_act(re_act<1) = 1;
        re_act(re_act>1000) = 1000;
        activity = kn_c((round(re_act))) ./ plant_scale;
        
    case 'high-pass'
        kn = TNC_CreateGaussian(500,150,1000,1);
        kn_c = cumsum(kn);
        re_act = activity.*1000;
        re_act(re_act<1) = 1;
        re_act(re_act>1000) = 1000;
        activity = kn_c((round(re_act)));

        activity = conv(activity,filter,'same');
        
end


switch option

    case 'poiss_simple'
        norm_activity = activity; % not sure I should norm this actually, but if you do: ./ max(abs(activity));

        % tmp = ones(20,1) * (1+randn(1,size(activity,2)/20)./2);
        % tmp = ones(20,1) * rand(1,size(activity,2)/20);
        tmp = 0.8+(randn(1,size(activity,2))/7);
        tmp2 = reshape(tmp,[1,numel(tmp)]);

        if size(tmp2,2)==size(activity,2)
            rand_seed = tmp2;
        else
            rand_seed = zeros(1,size(activity,2));
            rand_seed(1:size(tmp2,2)) = tmp2;
        end

        checks_raw = rand_seed<norm_activity;
        checks = find( [0 diff(checks_raw)] == 1 );

        if plotFlag
            figure(700); clf;
            plot(activity); hold on; plot(rand_seed,'k','color',[0.5 0.5 0.5]); 
            if numel(checks)>0
                plot(checks,zeros(1,numel(checks)),'r*');
            end
        end
        
    case 'state_simple'
        
        state           = zeros(1,numel(activity));
        checks_tmp      = zeros(1,numel(activity));
        norm_activity   = zeros(1,numel(activity));
%         back_p          = exp(([1:numel(activity)]-numel(activity)-3)./150);
        back_p          = exp(([1:numel(activity)]-numel(activity)-3)./200);
%         back_p          = exp(([1:numel(activity)]-numel(activity)-3)./25);
        lick_template   = zeros(1,numel(activity));
        lick_template(1:150:numel(activity)) = 1;
        
        norm_activity     = activity + back_p;
        rand_chks          = rand(1,numel(activity));
        
        reward=[1*ones(1,1600) 100*ones(1,1400)];
            
        for pp=1:numel(activity)-1
            
            act_p = norm_activity(pp);
            p_trans = act_p;

            if state(pp)==0
                if rand_chks(pp)< p_trans
                    state(pp+1)=1;
                end
            else
                if rand_chks(pp)< 0.005 / reward(pp)
                    state(pp+1)=0;
                else
                    state(pp+1)=1;
                end
            end            
        end
        
        tmp     = find([0 diff(state)]==1);
        tmp_neg = find([0 diff(state)]==-1);
        if numel(tmp)>0
            if numel(tmp)>numel(tmp_neg)
                tmp_neg = [tmp_neg numel(activity)];
            end
            for kk=1:numel(tmp)
                if numel([tmp(kk):tmp_neg(kk)]) > 100 | tmp(kk)>numel(activity)-100
%                     offset = round(10*rand(1));
                    offset = randperm(150,1);
                    if tmp_neg(kk)+offset > numel(activity)
                        checks_tmp(tmp(kk)+offset:numel(activity)) = lick_template(1:numel([tmp(kk)+offset:numel(activity)]));
                    else
                        checks_tmp(tmp(kk)+offset:tmp_neg(kk)+offset) = lick_template(1:numel([tmp(kk):tmp_neg(kk)]));
                    end
                end
            end
        end
        
        checks2 = find(checks_tmp==1) + reaction_time;
        checks = checks2(find(checks2<size(activity,2))); % there is always a pause right at water delivery

        if plotFlag
            figure(700); clf;
            hold off; plot(norm_activity+back_p); hold on; plot(state); plot(checks,ones(1,numel(checks)),'k*');
        end
        
end

varargout{1}=state;
        
