function [critic] = dlRNN_criticEngine(critic,stim_cond)

critic.et = zeros(numel(critic.w),1);

for t = 1:critic.steps-1

    critic.v(t) = sum(critic.w .* critic.x(:,t));
    critic.v(t+1) = sum(critic.w .* critic.x(:,t+1));

    % where d is TD error, r is reward, v is state value. Next, eligibility traces were updated by
    if stim_cond>=20
        critic.d(t) = (critic.r(t)*4) + critic.gamma*(critic.v(t+1) - critic.v(t));
    else
        critic.d(t) = critic.r(t) + critic.gamma*(critic.v(t+1) - critic.v(t));
    end

    % where et is eligibility traces for all the states, γ is a discounting factor from 0 to 1, λ is a constant to determine an updating rule and α is a learning rate. 
    critic.et = (critic.gamma * critic.lambda * critic.et) + (critic.alpha * critic.x(:,t));

    % Then, weights are updated by
    critic.w = critic.w + (critic.d(t) * critic.et);

end

critic.rpe_rew = critic.d(critic.rewTime);
critic.rpe_cue = critic.d(critic.cueTime);
