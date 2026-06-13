import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical

from agents.base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=128):
        super().__init__()
        self.shared=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim) ,
            nn.ReLU()
        )
        self.actor=nn.Linear(hidden_dim,action_dim)
        self.critic=nn.Linear(hidden_dim,1)

    def forward(self,state):
        features=self.shared(state)
        logits=self.actor(features)
        value=self.critic(features)
        return logits,value
    

class PPO(BaseAgent):
    def __init__(
        self,
        state_dim=2,
        action_dim=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        update_epochs=10,
        rollout_size=256,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5
    ):

        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.gamma=gamma
        self.clip_eps=clip_eps
        self.update_epochs=update_epochs
        self.rollout_size=rollout_size
        self.entropy_coef=entropy_coef
        self.value_coef=value_coef
        self.gae_lambda=gae_lambda
        self.max_grad_norm=max_grad_norm

        self.network=ActorCritic(state_dim,action_dim).to(self.device)

        self.optimizer=optim.Adam(self.network.parameters(),lr=lr)
        self.training_mode=True

        self.reset_buffer()

        self.prev_state=None
        self.prev_action=None
        self.prev_log_prob=None
        self.prev_value=None
    

    def train_mode(self):
        self.training_mode=True
        self.network.train()

    def eval_mode(self):
        self.training_mode=False
        self.network.eval()

    def reset_buffer(self):
        self.states=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.log_probs=[]
        self.values=[]

    def take_action(self, state):
        
        state_tensor=torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits,value=self.network(state_tensor)
            dist=Categorical(logits=logits)

            if self.training_mode:
                
                action=dist.sample()
            else:
                action=torch.argmax(logits,dim=-1)

            log_prob=dist.log_prob(action)

            self.prev_state=state
            self.prev_action=action.item()
            self.prev_log_prob=log_prob.item()
            self.prev_value=value.item()

        return self.prev_action
    
    def update(self,state,reward,action,done=False):
        if self.prev_state is None:
            return
        
        self.states.append(self.prev_state)
        self.actions.append(self.prev_action)
        self.rewards.append(reward)
        self.log_probs.append(self.prev_log_prob)
        self.values.append(self.prev_value)
        self.dones.append(done)


        if len(self.states)>=self.rollout_size:
            self.learn()
    

    def compute_gae(self):
        rewards=np.array(self.rewards)
        values=np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32) 
        advantages=np.zeros_like(rewards)

        gae=0
        next_value=0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t] 

            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae 

            advantages[t]=gae
            next_value=values[t]
        returns=advantages+values

        return advantages,returns
    
    def learn(self):

        if len(self.states)==0:
            return
        advantages,returns=self.compute_gae()
        
        states=torch.FloatTensor(np.array(self.states)).to(self.device)
        actions=torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs=torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages=torch.FloatTensor(advantages).to(self.device)
        returns=torch.FloatTensor(returns).to(self.device)
        
        if len(advantages)>1:
            advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)

        for _ in range(self.update_epochs):
            logits,values=self.network(states)

            dist=Categorical(logits=logits)

            new_log_probs=dist.log_prob(actions)
            entropy=dist.entropy().mean()
            
            ratio=torch.exp(new_log_probs-old_log_probs)
            surr1=ratio*advantages
            surr2=torch.clamp(ratio,1-self.clip_eps,1+self.clip_eps)*advantages

            actor_loss=-torch.min(surr1,surr2).mean()
            
            critic_loss=nn.functional.mse_loss(values.squeeze(),returns)

            loss=(actor_loss+self.value_coef*critic_loss -self.entropy_coef*entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(),self.max_grad_norm)
            self.optimizer.step()

        self.reset_buffer()









        