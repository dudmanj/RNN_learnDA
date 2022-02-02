%% Initialize the critic
critic.rewTime = round(find( [0 diff(curr_input(2,:))]>0 , 1 ) / 100);
critic.cueTime = ceil(find( [0 diff(curr_input(1,:))]>0 , 1 ) / 100) + 1;
critic.steps = size(curr_input,2) / 100; % 100 ms long boxcar basis set
critic.rpe_rew = 0;
critic.rpe_cue = 0;
critic.w = zeros(critic.steps,1);
critic.x = zeros(numel(critic.w),critic.steps);
            
for p=critic.cueTime+1:critic.steps
    critic.x(p,p) = 1;
end

critic.r = zeros(1,critic.steps);
critic.d = zeros(1,critic.steps);
critic.r(critic.rewTime) = 1;
critic.v = zeros(1,critic.steps);
critic.alpha = 0.001;
critic.lambda = 1;
critic.gamma = 1;
critic


%% RUN the value model 
stim_cond = 0;
curr_rpe = [];
for reps = 1:800

    [critic] = dlRNN_criticEngine(critic,stim_cond);
    curr_rpe(reps,:) = critic.d;
end

figure(1); clf;
imagesc(curr_rpe);
