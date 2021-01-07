function [checks,varargout] = dlRNN_Pcheck_transfer(activity)

option = 'state_simple';
plotFlag = 0;
reaction_time = 45;

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
        back_p          = exp(([1:numel(activity)]-numel(activity)-3)./50);
        lick_template   = zeros(1,numel(activity));
        lick_template(1:120:numel(activity)) = 1;
        
%         norm_activity(activity>0)   = activity(activity>0);
%         norm_activity               = norm_activity + back_p;
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
                    offset = round(10*rand(1));
                    if tmp_neg(kk)+offset > numel(activity)
                        checks_tmp(tmp(kk)+offset:numel(activity)) = lick_template(1:numel([tmp(kk)+offset:numel(activity)]));
                    else
                        checks_tmp(tmp(kk)+offset:tmp_neg(kk)+offset) = lick_template(1:numel([tmp(kk):tmp_neg(kk)]));
                    end
                end
            end
        end
        
        checks2 = find(checks_tmp==1) + reaction_time;
        checks = checks2(find(checks2<size(activity,2)));

        if plotFlag
            figure(700); clf;
            hold off; plot(norm_activity+back_p); hold on; plot(state); plot(checks,ones(1,numel(checks)),'k*');
        end
        
end

varargout{1}=state;
        
%% USEFUL FOR TESTING SETTINGS
% for pp=1:100
%    [checks] = dlRNN_Pcheck_transfer(ones(1,1000));
%    hmm(pp) = min(checks);
% end
% 
% mean(hmm)
% std(hmm)