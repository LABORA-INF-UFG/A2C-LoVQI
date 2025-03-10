import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
from numpy import random
from ns3gym import ns3env  # Ambiente NS-3
from A2C_mapping import Actor, Critic, save_checkpoint, load_checkpoint, save_metrics

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print(f"Executando treinamento no dispositivo: {device}")

# Create a parser
parser = argparse.ArgumentParser(description='A2C for n UAVs')
# Add arguments to the parser
parser.add_argument('--v', type=int, help='Verbose mode')
parser.add_argument('--pr', type=int, help='Port number')
parser.add_argument('--gr', type=int, help='Grid dimension (gr x gr)')
parser.add_argument('--sz', type=int, help='Area side size')
parser.add_argument('--dv', type=int, help='Number of Devices')
parser.add_argument('--gw', type=int, help='Number of Gateways')
parser.add_argument('--ep', type=int, help='Number of episodes')
parser.add_argument('--st', type=int, help='Number of steps')
parser.add_argument('--ss', type=int, help='Start NS-3 Simulation')
parser.add_argument('--so', type=int, help='Start Optimal')
parser.add_argument('--sd', type=int, help='Seed number')
parser.add_argument('--out', type=int, help='Plot the results')
args = parser.parse_args()

# Parâmetros e configurações do A2C
port = args.pr
dim_grid = args.gr  # Tamanho da grade
area_side = args.sz  # Lado da área de simulação
nDevices = args.dv  # Número de dispositivos
nGateways = args.gw  # Número de gateways
simTime = 100  # seconds
stepTime = 25  # seconds
verbose = False if args.v == 0 else True  # Verbosidade do ambiente
debug = False  # Depuração do ambiente
plot_data = True  # Gerar gráficos
start_sim = args.ss
start_optimal = args.so
debug = 0 if start_sim == 1 else 1  # Iniciar simulação em condições ótimas
episodes = args.ep  # Número de episódios
steps_per_episode = args.st  # Passos por episódio
virtual_positions = dim_grid * dim_grid
step_size = area_side / dim_grid  # Tamanho do passo
print_movements = True
sim_seed = args.sd  # Semente de simulação
run_seed = episodes - 1
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--simSeed": sim_seed,
           "--runSeed": run_seed,
           "--virtualPositions": virtual_positions,
           "--startOptimal": start_optimal,
           "--areaSide": area_side}

movements = ['up', 'right', 'down', 'left', 'stay']
# Caminho para salvar arquivos
path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"
checkpoint_filename = f"{CHECKPOINT_DIR}A2C_chkpt_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.pth"
best_positions_file_name = f"{path_files}A2C_best_positions_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
results_file_name = f"{path_files}A2C_results_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
movements_file_name = f"{path_files}A2C_movements_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat.dat"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

action_space = list(itertools.product(movements, repeat=nGateways))  # Espaço de ações
area = dim_grid * dim_grid
state_size = nGateways * 3

# Inicializar o ambiente NS-3
ns3_env = ns3env.Ns3Env(
    port=port,
    startSim=args.ss,
    simSeed=sim_seed,
    simArgs=simArgs,
    debug=debug,
    spath="/home/rogerio/git/ns-allinone-3.42/ns-3.42/"
)

# Configurações
n_actions = len(action_space)
actor = Actor(state_size, n_actions).to(device)
critic = Critic(state_size).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
gamma = 0.92
entropy_coeff = 0.05
max_noise = 1.0
alpha = 0.995


# Armazenamento de dados
episode_mean_actor_loss = [0]
episode_mean_critic_loss = [0]
episode_mean_rewards = [0]
episode_qualified_mean_rewards = [0]
execution_times = []
episodes_visited_states = {}
improvements = []
# map de episodes_movements
episodes_movements = {}

k = 0
kq = 0
done = False
allowed_checkpoint = int(0.2 * episodes)
# Carrega os pesos da rede
if os.path.exists(checkpoint_filename):
    load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, checkpoint_filename)

# Se foi interrompido, carrega o último episódio executado
if os.path.exists(results_file_name):
    data = pd.read_csv(results_file_name)
    initial_episode = data['episodio'].max() + 1
    final_episode = episodes + initial_episode
else: # Primeira execução
    initial_episode = 1
    final_episode = episodes

best_reward = -np.inf

for episode in range(initial_episode, final_episode + 1):
    start = time.time()
    state = ns3_env.reset()
    visited_states = [state]
    episode_movements = []
    rewards = []
    q_rewards = []
    actor_losses = []
    critic_losses = []

    # Controle de temperatura e ruído
    temperature = max(0.1, 1.0 - (episode / episodes))  # Decaimento linear
    noise_scale = max_noise * (1 - episode / episodes)


    step = 0
    if done:  # Colision in last episode - Reset()
        obs = ns3_env.reset(sim_seed)
    else:
        state, reward, _, _ = ns3_env.get_state()
    initial_episode_reward = reward

    while step < steps_per_episode:
        # if step % 100 == 0:
        #     print(f'Step {step}')
        # Converte o estado para tensor e move para o dispositivo
        state, _, _, _ = ns3_env.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        # Computa a probabilidade usando o ator
        # Amostras usando temperatura - Política estocástica
        probs = actor(state_tensor) / temperature
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        i_action = int(action.item())
        # Executa a ação e obtém o próximo estado e recompensa
        if i_action >= len(action_space) or i_action < 0:
            print(f"Erro ao executar choose_action com iAction={action}: iAction >= len(action_space)")
        else:
            try:
                next_state, reward, done, info = ns3_env.step(i_action)
            except Exception as e:
                print(f"Erro ao executar ns3_env.step com iAction={i_action} e action:{action}: {e}")
                raise
        if reward > best_reward:
            best_reward = reward
            best_positions = next_state
            best_info = info
            best_episode = episode
            best_step = step
        # Recompensa por evitar estados inválidos
        if info and "Collision" not in info and "OutOfArea" not in info:
            reward += 0.1
        # Converte o próximo estado para tensor e move para o dispositivo
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

        # Calcula a vantagem
        advantage = reward + (1 - done) * gamma * critic(next_state_tensor) - critic(state_tensor)
        advantage = advantage.to(device)

        visited_states.append(next_state)
        if print_movements:
            str_state = str(state).replace("[ ", "").replace("\n", "").replace("\t", "").replace("    ", ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            str_next_state = str(next_state).replace("[ ", "").replace("\n", "").replace("\t", "").replace("    ",
                                                                                                           ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            episode_movements.append((step, str_state, str_next_state, reward, info, action, step_size))

        if verbose:
            print(
                f'Step: {step} Rw:{reward} I: + {info}')  # Ac:{action} St:{next_state}')
        # Certifique-se de que a recompensa e a vantagem estão no mesmo dispositivo
        reward = torch.tensor(reward, dtype=torch.float32).to(device)

        # Calcula e otimiza a perda do crítico
        critic_loss = advantage.pow(2).mean()
        critic_losses.append(critic_loss.detach().cpu().numpy())
        # Calcula e otimiza a perda do ator
        actor_loss = -(dist.log_prob(action) * advantage.detach())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropy_loss = -entropy_coeff * dist.entropy().mean()
        actor_loss = actor_loss + entropy_loss

        # Backpropagation e atualização dos pesos
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        rewards.append(reward.item())  # Converte para item numérico e adiciona à lista
        if info and "Collision" not in info and "OutOfArea" not in info:
            q_rewards.append(reward.item())
        step += 1
        state = next_state

    episodes_visited_states[episode] = visited_states

    previous_mean_reward = episode_mean_rewards[-1] if len(episode_mean_rewards) > 0 else 0
    previous_mean_critic_loss = episode_mean_critic_loss[-1] if len(episode_mean_critic_loss) > 0 else 0
    previous_mean_actor_loss = episode_mean_actor_loss[-1] if len(episode_mean_actor_loss) > 0 else 0
    previous_qualified_mean_reward = episode_qualified_mean_rewards[-1] if len(
        episode_qualified_mean_rewards) > 0 else 0

    # Calcula as médias dos registros
    mean_reward = ((k + 1) * previous_mean_reward + np.mean(rewards)) / (k + 2)
    episode_mean_rewards.append(mean_reward)

    if info and "Collision" not in info and "OutOfArea" not in info:
        qualified_mean_reward = ((kq + 1) * previous_qualified_mean_reward + np.mean(q_rewards)) / (kq + 2)
        kq+=1
        episode_qualified_mean_rewards.append(qualified_mean_reward)
    else:
        qualified_mean_reward=0

    mean_actor_loss = ((k + 1) * previous_mean_actor_loss + np.mean(actor_losses)) / (k + 2)
    episode_mean_actor_loss.append(mean_actor_loss)

    mean_critic_loss = ((k + 1) * previous_mean_critic_loss + np.mean(critic_losses)) / (k + 2)
    episode_mean_critic_loss.append(mean_critic_loss)

    k += 1

    if print_movements:
        episodes_movements[episode] = episode_movements
    final_episode_reward = reward
    stop_time = time.time()

    improvements.append(1 if ((final_episode_reward > initial_episode_reward) == True) else 0)

    episode_time = time.time() - start
    execution_times.append(episode_time)
    print(f'Episode {episode}/{episodes} '
          f'[{'+' if ((final_episode_reward > initial_episode_reward) == True) else '-'}'
          f'] Time elapsed: {episode_time}'
          f' Mean reward: {mean_reward:.4f}'
          f' Qualified Mean reward: {qualified_mean_reward:.4f}'
          f' Mean Actor Loss: {mean_actor_loss:.4f}'
          f' Critic Loss: {mean_critic_loss:.4f}\n')

    # Verifica se é hora de salvar o modelo e as métricas
    if episode % allowed_checkpoint == 0:
        save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, checkpoint_filename)
        save_metrics(results_file_name, best_positions_file_name, movements_file_name, execution_times,
                     episode_mean_rewards, episode_mean_actor_loss, episode_mean_critic_loss, best_episode, best_step,
                     best_reward, best_positions, best_info, print_movements, episodes_movements, "a")

print(f'For {episodes} episodes, there was {sum(improvements)} improvements '
      f'({round(sum(improvements) * 100 / episodes, 2)}%) and '
      f'{episodes - sum(improvements)} worse results ('
      f'{round((episodes - sum(improvements)) * 100 / episodes, 2)}%)')
# Salvar modelo final
save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, checkpoint_filename)
# Salvar métricas finais
save_metrics(results_file_name, best_positions_file_name, movements_file_name, execution_times,
             episode_mean_rewards, episode_mean_actor_loss, episode_mean_critic_loss, best_episode, best_step,
             best_reward, best_positions, best_info, print_movements, episodes_movements, "w")
print("Treinamento concluído!")
ns3_env.close()
