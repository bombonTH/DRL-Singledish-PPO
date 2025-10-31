import os
import random
import tyro
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm, trange

import single_dish_env_image_obs


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""

    # Algorithm specific arguments
    env_id: str = "SingleDish-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-5
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    max_steps: int = 4096
    """Maximum number of steps to run per episode"""
    resume: bool = False
    """whether to continue training"""
    fix_std: bool = False
    """whether to use fixed std"""
    std: float = 0.25
    """whether to use fixed std"""
    gae: bool = True
    """Whether to use gae"""
    training: bool = True
    """Toggle training/validating"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, run_name, gamma):
    def thunk():
        env = gym.make(args.env_id,
            max_steps=args.max_steps,
            random_init_step=True,
            aps=1,
            rank=idx,
            dilation=1,
            random=True,
            random_mag=0.2,
            hit_margin=0.01,
            hit_required=10,
            num_clusters= 1 + (idx % 4),
            num_obstacles = int(idx % 4),
            default_reward=0,
            empty_sky_final = True,                                   
            show=idx<4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.update_count = 0
        self.noise_prob = Normal(0, args.std)
        self.noise = self.noise_prob.sample().to(device).detach()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1), # 32x64x64
            nn.ReLU(),
                        
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1), # 64x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x32x32
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1), # 128x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x16x16
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1), # 256x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256x8x8
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1), # 256x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256x4x4
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, padding=0, stride=1, groups=1), # 256x1x1
            nn.Flatten(),
        )
        
        self.bore_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1), # 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #32x32x32
                        
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1), # 64x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #64x16x16
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1), # 128x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x8x8
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1), # 256x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256x4x4
           
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, padding=0, stride=1, groups=1), # 256x1x1
            nn.Flatten(),
        )
        
        self.lstm = nn.LSTM(256, 256)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.latent_feature = nn.Sequential(
            layer_init(nn.Linear(256+256+7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        
        self.actor_logstd = nn.Sequential(
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.latent_feature(torch.cat((self.encoder(x[:,:3]), self.bore_encoder(x[:,3:4]), x[:,0,:7,0]), 1))
        
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
        
    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        self.update_count += 1
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        
        if self.update_count % 128 == 0 or self.noise.shape != action_mean.shape:
            self.update_count = 0
            noise_index = np.arange(4, 16)
            np.random.shuffle(noise_index)
            self.noise = self.noise_prob.sample(action_mean.shape).to(device).detach()
            
            # Uncomment to enable different noise scale for each of the envs. Assume 16 envs.
            # self.noise[noise_index[:4]] = self.noise[noise_index[:4]] * 0.1
            # self.noise[noise_index[4:8]] = self.noise[noise_index[4:8]] * 0.5
            # self.noise[noise_index[8:]] = self.noise[noise_index[8:]] * 2
            # self.noise[:4] = 0
            
        if args.fix_std:
            action_std = torch.full_like(action_mean, args.std).to(device)
        else:
            action_logstd = self.actor_logstd(latent_action.detach())
            action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + self.noise
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state
        


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    scaler = GradScaler()
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    
    if args.resume:
        state_dict_agent = torch.load(os.path.join(os.getcwd(), 'weights','agent.pt'), weights_only=True)
        state_dict_optimizer = torch.load(os.path.join(os.getcwd(), 'weights','optimizer.pt'), weights_only=True)
        state_dict_scaler = torch.load(os.path.join(os.getcwd(), 'weights','scaler.pt'), weights_only=True)
        agent.load_state_dict(state_dict_agent, strict=True)
        optimizer.load_state_dict(state_dict_optimizer)
        scaler.load_state_dict(state_dict_scaler)
    else:
        state_dict_encoder = torch.load(os.path.join(os.getcwd(), 'weights','encoder.pt'), weights_only=True)
        agent.load_state_dict(state_dict_encoder['extractor'], strict=False)
        
    init_step = 0e6

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )
    
    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            _next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(_next_done).to(device)
            
            if any(_next_done):
                episodic_return = np.mean(infos['cum_reward'][_next_done])
                done_count = np.mean(infos['done'][_next_done])
                episodic_length = np.mean(infos['num_step'][_next_done] - infos['init_step'][_next_done])
                average_return = episodic_return / episodic_length
                done_per_step = done_count / episodic_length
                sq_deg_per_hour = done_per_step * 11309.0673773
                writer.add_scalar("charts/episodic_return", episodic_return, init_step + global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, init_step + global_step)
                writer.add_scalar("charts/step_return", average_return, init_step + global_step)
                writer.add_scalar("charts/Sq.deg_per_hour", sq_deg_per_hour, init_step + global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                with autocast(device_type='cuda', dtype=torch.float16) and torch.no_grad():
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    
                    if args.training:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], init_step + global_step)
        writer.add_scalar("losses/epochs", epoch, init_step + global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), init_step + global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), init_step + global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), init_step + global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), init_step + global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), init_step + global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), init_step + global_step)
        writer.add_scalar("losses/explained_variance", explained_var, init_step + global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), init_step + global_step)

    if args.save_model:
        agent_path = f"runs/{run_name}/agent.pt"
        optimizer_path = f"runs/{run_name}/optimizer.pt"
        scaler_path = f"runs/{run_name}/scaler.pt"
        torch.save(agent.state_dict(), agent_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        torch.save(scaler.state_dict(), scaler_path)
        print(f"model saved to {agent_path}")

    envs.close()
    writer.close()
