# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from ast import expr_context
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward, NormalizeObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from pathlib import Path
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# lsy drone racing environment
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.reinforcement_learning.rl_env_wrapper import RLDroneRacingWrapper
from lsy_drone_racing.utils import load_config


@dataclass
# region Args
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    start_from_scratch: bool = True
    """start from scratch or load from latest checkpoint"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-drone-racing"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "DroneRacing-v0"
    """the id of the environment"""
    total_timesteps: int = int(4e6)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    dev_envs: str = "gpu"
    """run jax envrionments on cpu/gpu"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1024
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # region Reward Coef
    k_alive:        float = 0.5
    k_alive_anneal: float = 1.0 # anneal alive reward at every step
    k_obst:         float = 0.0
    k_obst_d:       float = 0.0
    k_gates:        float = 4.0
    k_center:       float = 0.3
    k_center_d:     float = 0.1
    k_vel:          float = -0.0
    k_act:          float = 0.01
    k_act_d:        float = 0.001
    k_yaw:          float = 0.1
    k_crash:        float = 25.0
    k_success:      float = 100.0
    k_finish:       float = 40.0
    k_imit:         float = 1.0
    """REWARD PARAMETERS"""

# load model
def load_latest_model(log_dir: Path) -> Path:
    import re
    patt = re.compile(r"rl_drone_racing_(\d+)\.pth$")
    candidates = [(int(m.group(1)), f) for f in log_dir.glob("rl_drone_racing_*.pth")
                  if (m := patt.match(f.name))]
    if not candidates:
        raise FileNotFoundError("No saved model found in log_dir")
    latest_path = max(candidates, key=lambda t: t[0])[1]
    print(f"Loading model: {latest_path}")
    return latest_path

# region Env
def make_env(config, args, gamma):
    env = VecDroneRaceEnv(
        num_envs       = args.num_envs,
        freq           = config.env.freq,
        sim_config     = config.sim,
        track          = config.env.track,
        sensor_range   = config.env.sensor_range,
        control_mode   = config.env.control_mode,
        disturbances   = config.env.get("disturbances"),
        randomizations = config.env.get("randomizations"),
        seed           = config.env.seed,
        device         = args.dev_envs,
    )
    env = JaxToNumpy(env)
    env = RLDroneRacingWrapper(
        env,
        k_alive    = args.k_alive,
        k_alive_anneal  = Args.k_alive_anneal,
        k_obst     = args.k_obst,
        k_obst_d   = args.k_obst_d,
        k_gates    = args.k_gates,
        k_center   = args.k_center,
        k_center_d = args.k_center_d,
        k_vel      = args.k_vel,
        k_act      = args.k_act,
        k_act_d    = args.k_act_d,
        k_yaw      = args.k_yaw,
        k_crash    = args.k_crash,
        k_success  = args.k_success,
        k_finish   = args.k_finish,
        k_imit     = args.k_imit,
    ) # my custom wrapper
    env = RecordEpisodeStatistics(env) # for wandb log
    env = NormalizeReward(env, gamma=gamma) # might help
    env = NormalizeObservation(env)
    # env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# region Agent
class Agent(nn.Module):
    """the best drone flyer"""
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        obs_dim = np.prod(envs.single_observation_space.shape)
        act_dim = np.prod(envs.single_action_space.shape)

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        # Actor: shared base, two output heads
        self.actor_base = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(128, act_dim), std=0.01)
        # self.actor_logstd = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        base = self.actor_base(x)
        mean = self.actor_mean(base)
        # log_std = torch.clamp(self.actor_logstd(base), -5.0, 2.0) # important for training stability
        action_logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(action_logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, torch.as_tensor(self.envs.single_action_space.low, device=action.device), torch.as_tensor(self.envs.single_action_space.high, device=action.device))        
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)
    
    @torch.no_grad()
    def act(self, x, deterministic=True):
        base = self.actor_base(x)
        mean = self.actor_mean(base)
        if deterministic:
            return mean
        action_logstd = self.actor_logstd.expand_as(mean)
        std  = torch.exp(action_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, torch.as_tensor(self.envs.single_action_space.low, device=action.device), torch.as_tensor(self.envs.single_action_space.high, device=action.device))
        return dist.sample()

# region Main
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"rl_drone_racing_{int(time.time())}"
    config = load_config(Path(__file__).parents[2] / "config/trainrl.toml")
    args.env_id = config.env.id # synchronize environment
    args.seed = config.env.seed # synchronize seed
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
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
    envs = make_env(config, args, args.gamma)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # region Load Model
    log_dir = Path(__file__).parent / "log4"
    if args.start_from_scratch == False:
        model_path = load_latest_model(log_dir)
        agent.load_state_dict(torch.load(model_path))

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

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # region Rollout
        num_finishes = 0.0
        ep_return = 0.0
        ep_length = 0.0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                # envs.render()
            if "episode" in infos:
                ep_return += np.sum(infos['episode']['r'][infos['_episode']])
                ep_length += np.sum(infos['episode']['l'][infos['_episode']])
                num_finishes += np.sum(infos['_episode'])
        
        # write summary at each iteration
        print(f"total_iter={iteration}, global_step={global_step}, "
              f"ep_return={ep_return/num_finishes if num_finishes > 0 else 0.0}, "
              f"ep_length={ep_length/num_finishes if num_finishes > 0 else 0.0} "
              )
        writer.add_scalar("charts/episodic_return", ep_return/num_finishes if num_finishes > 0 else 0.0, global_step)
        writer.add_scalar("charts/episodic_length", ep_length/num_finishes if num_finishes > 0 else 0.0, global_step)

        # bootstrap value if not done
        # region GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # region NN update
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and iteration%5 == 0:
            log_dir = Path(__file__).parent / "log4"
            model_path = log_dir / f"rl_drone_racing_iter_{iteration}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        
    if args.save_model:
        import re
        log_dir = Path(__file__).parent / "log4"
        log_dir.mkdir(parents=True, exist_ok=True)

        pattern = re.compile(rf"rl_drone_racing_(\d+)\.pth$")
        existing = [
            (int(m.group(1)), f)
            for f in log_dir.glob(f"rl_drone_racing_*.pth")
            if (m := pattern.match(f.name))
        ]
        next_idx = (max(idx for idx, _ in existing) + 1) if existing else 0
        
        final_path = log_dir / f"rl_drone_racing_{next_idx}.pth"
        torch.save(agent.state_dict(), final_path)
        print(f"✅ Final model saved to {final_path}")
    

    envs.close()
    writer.close()
