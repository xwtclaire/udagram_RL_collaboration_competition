# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from collections import namedtuple, deque
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPDATE_EVERY = 1        # how often to update the network
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.01, random_seed=0):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(24, 52, 2, random_seed), 
                             DDPGAgent(24, 52, 2, random_seed)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=True):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        state_full = np.reshape(states, newshape=(-1))
        next_state_full = np.reshape(next_states, newshape=(-1))
        action_full = np.reshape(actions, newshape=(-1))
                                
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, state_full, action, action_full, reward, next_state, next_state_full, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for a_i in range(2):
                    experiences = self.memory.sample()
                    self.update(experiences, a_i)
                self.update_targets()

    def update(self, experiences, agent_number):
        """update the critics and actors of all the agents """

        state, state_full, actions, actions_full, rewards, next_state, next_state_full, dones = experiences
        
        obs = state
        next_obs = next_state
        obs_full = state_full
        next_obs_full = next_state_full

        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        #target_actions = self.target_act(next_obs)
        #target_actions = agent.actor_target(next_obs)
        target_actions = [agent.actor_target(next_obs_full[:,0:int(next_obs_full.shape[1]/2)]),
                          agent.actor_target(next_obs_full[:,int(next_obs_full.shape[1]/2):])]
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.critic_target(target_critic_input)
        
        # Compute Q targets for current states (y_i)
        y = rewards + (self.discount_factor * q_next * (1 - dones))
        
        critic_input = torch.cat((obs_full, actions_full), dim=1).to(device)
        q = agent.critic_local(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = agent.actor_local(obs)
        q_input = [self.maddpg_agent[0].actor_local(obs_full[:,0:int(obs_full.shape[1]/2)]),
                   self.maddpg_agent[1].actor_local(obs_full[:,int(obs_full.shape[1]/2):])]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic_local(q_input2).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            self.soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
            
    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for updatenumpy.concatenate
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_full", "action", "action_full", "reward", "next_state", "next_state_full", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, state_full, action, action_full, reward, next_state,next_state_full, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_full, action, action_full, reward, next_state, next_state_full, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states_full = torch.from_numpy(np.vstack([e.state_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        actions_full = torch.from_numpy(np.vstack([e.action_full for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states_full = torch.from_numpy(np.vstack([e.next_state_full for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, states_full, actions, actions_full, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
            
            
            




