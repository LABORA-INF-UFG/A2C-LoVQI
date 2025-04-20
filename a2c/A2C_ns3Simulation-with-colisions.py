import argparse
import itertools
import os
import time
from math import isnan

import numpy as np
import pandas as pd
import torch
from ns3gym import ns3env  # Ambiente NS-3
from A2C_mapping import Actor, Critic, RewardNormalizer, save_checkpoint, load_checkpoint, save_metrics

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
n_episodes = args.ep  # Número de episódios
steps_per_episode = args.st  # Passos por episódio
virtual_positions = dim_grid * dim_grid
step_size = area_side / dim_grid  # Tamanho do passo
print_movements = True
sim_seed = args.sd

simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--virtualPositions": virtual_positions,
           "--startOptimal": start_optimal,
           "--areaSide": area_side}

movements = ['up', 'right', 'down', 'left', 'stay']
# Caminho para salvar arquivos
# path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
# CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"
# checkpoint_filename = f"{CHECKPOINT_DIR}A2C_chkpt_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.pth"
# best_positions_file_name = f"{path_files}A2C_best_positions_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
# results_file_name = f"{path_files}A2C_results_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
# movements_file_name = f"{path_files}A2C_movements_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat.dat"
path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/checkpoints/"
checkpoint_filename = f"{CHECKPOINT_DIR}A2C_chkpt_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}SS.pth"
best_positions_file_name = f"{path_files}A2C_best_positions_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}SS.dat"
results_file_name = f"{path_files}A2C_results_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}SS.dat"
movements_file_name = f"{path_files}A2C_movements_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}SS.dat"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

action_space = list(itertools.product(movements, repeat=nGateways))  # Espaço de ações
area = dim_grid * dim_grid
state_size = nGateways * 3

# Inicializar o ambiente NS-3
ns3_env = ns3env.Ns3Env(
    port=port,
    startSim=start_sim,
    simSeed=sim_seed,
    simArgs=simArgs,
    debug=debug,
    spath="/home/rogerio/git/ns-allinone-3.42/ns-3.42/"
)

# Configurações as Redes do Ator e do Crítico
n_actions = len(action_space)
actor = Actor(state_size, n_actions).to(device)
critic = Critic(state_size).to(device)
# Parâmetros para o aprendizado
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1.5e-5, weight_decay=1e-5)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-5)
max_noise = 1.0
epsilon = 1e-8
gamma = 0.90  # Ajustado para evitar propagação excessiva de erro
alpha = 0.99  # Menos sensível a variações bruscas na normalização da recompensa
reward_normalizer = RewardNormalizer(alpha=alpha, epsilon=epsilon)
initial_entropy = 0.035  # Valor inicial
final_entropy = 0.01  # Valor final
entropy_decay = 0.99  # Taxa de decaimento
entropy_coeff = initial_entropy

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


# Se foi interrompido, carrega o último episódio executado
if os.path.exists(results_file_name):
    data = pd.read_csv(results_file_name)
    loaded_metrics = True
    # Carregar últimos resultados (ultima linha)
    last_row = data.iloc[-1]
    episodio, tempo, reward, q_reward, actor_loss, critic_loss = last_row['episodio'], last_row['tempo'], last_row[
        'reward'], last_row['q_reward'], last_row['actor_loss'], last_row['critic_loss']
    # Carregar todas as linhas nas respectivas listas
    # episodes = data['episodio'].tolist()
    # execution_times = data['tempo'].tolist()
    # episode_mean_rewards = data['reward'].tolist()
    # episode_qualified_mean_rewards = data['q_reward'].tolist()
    # episode_mean_actor_loss = data['actor_loss'].tolist()
    # episode_mean_critic_loss = data['critic_loss'].tolist()
    # Computar a sequencia de episódios à executar
    initial_episode = data['episodio'].max() + 1
    final_episode = n_episodes + initial_episode
else: # Primeira execução
    tempo = 0
    initial_episode = 1
    final_episode = n_episodes
    loaded_metrics = False

k = 0
done = False
allowed_checkpoint = int(0.2 * n_episodes)
allowed_checkpoint = 10 if 0 <= allowed_checkpoint < 10 else allowed_checkpoint
# allowed_checkpoint = final_episode if allowed_checkpoint > final_episode else allowed_checkpoint
# Carrega os pesos da rede
if os.path.exists(checkpoint_filename):
    load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, checkpoint_filename)

best_reward = -np.inf
out_of_area = 0
accumulated_out_of_area_penalty = 0

for episode in range(initial_episode, final_episode + 1):
    step = 0
    start = time.time()
    episode_movements = []
    rewards = []
    q_rewards = []
    actor_losses = []
    critic_losses = []

    state = ns3_env.reset(simSeed=sim_seed)
    _, reward, _, _ = ns3_env.get_state()
    visited_states = [state]

    # Controle de temperatura e ruído
    temperature = max(0.1, 1.0 - (episode / n_episodes))  # Decaimento linear
    noise_scale = max_noise * (1 - episode / n_episodes)
    initial_episode_reward = reward
    accumulated_out_of_area_penalty = 0

    while step < steps_per_episode:
        # Converte o estado para tensor e move para o dispositivo
        state, _, _, _ = ns3_env.get_state()
        # state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        state_tensor = (state_tensor - state_tensor.mean()) / (state_tensor.std() + 1e-8)

        # Computa a probabilidade usando o ator
        # Amostras usando temperatura - Política estocástica
        probs = actor(state_tensor) / temperature
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        i_action = int(action.item())
        # Executa a ação e obtém o próximo estado e recompensa
        try:
            next_state, reward, done, info = ns3_env.step(i_action)
        except Exception as e:
            print(f"Erro ao executar ns3_env.step com iAction={i_action} e action:{action}: {e}")
            raise
        # Tentativa de evitar múltiplas saídas de área simultâneas
        # Em caso de saída da área aumenta a penalidade proporcionalmente
        accumulated_out_of_area_penalty = accumulated_out_of_area_penalty + 0.3 if reward == -1 else 0.0
        reward -= accumulated_out_of_area_penalty
        q_reward = 0
        # Registra as recompensas qualificadas
        if info and "OutOfArea" not in info and "Collision" not in info:
            reward += 0.5
            q_rewards.append(reward)
            q_reward = reward

        # Registra o estado da melhor recompensa por episódio
        if reward > best_reward:
            best_reward = reward
            best_positions = next_state
            best_info = info
            best_episode = episode
            best_step = step
        # Normaliza a recompensa
        reward = reward_normalizer.normalize(reward)
        # Registra a recompensa do passo
        rewards.append(reward)
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

        # Calcula e otimiza a perda do crítico
        critic_loss = advantage.pow(2).mean()
        critic_losses.append(critic_loss.detach().cpu().numpy())
        # Calcula e otimiza a perda do ator
        actor_loss = -(dist.log_prob(action) * advantage.detach())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropy_loss = -entropy_coeff * dist.entropy().mean()
        actor_loss = actor_loss + entropy_loss

        entropy_coeff = max(final_entropy, initial_entropy * (entropy_decay ** episode))  # Decaimento suave

        # Evitar gradientes explosivos com clipping
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

        # Backpropagation e atualização dos pesos
        # actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        step += 1
        state = next_state
        # Em caso de colisão termina o episódio ou se o total de penalizações
        # por saídas de área extrapolar a penalização da colisão
        if done:
            break


    # Ajustar a taxa de exploração X explotação
    reward_normalizer.update_epsilon()

    # em caso de colisão ou excesso de saídas de área salta para o próximo episódio
    if "Collision" in info or accumulated_out_of_area_penalty < -2.0:
        continue

    episodes_visited_states[episode] = visited_states

    previous_mean_reward = episode_mean_rewards[-1] if len(episode_mean_rewards) > 0 else 0
    previous_mean_critic_loss = episode_mean_critic_loss[-1] if len(episode_mean_critic_loss) > 0 else 0
    previous_mean_actor_loss = episode_mean_actor_loss[-1] if len(episode_mean_actor_loss) > 0 else 0
    previous_qualified_mean_reward = episode_qualified_mean_rewards[-1] if len(
        episode_qualified_mean_rewards) > 0 else 0

    # Calcula as médias dos registros
    mean_reward = ((k + 1) * previous_mean_reward + np.mean(rewards)) / (k + 2)
    episode_mean_rewards.append(mean_reward if not isnan(mean_reward) else 0)

    improvements.append(1 if mean_reward > previous_mean_reward else 0)

    qualified_mean_reward = ((k + 1) * previous_qualified_mean_reward + np.mean(q_rewards)) / (k + 2)
    episode_qualified_mean_rewards.append(qualified_mean_reward if not isnan(qualified_mean_reward) else 0)

    mean_actor_loss = ((k + 1) * previous_mean_actor_loss + np.mean(actor_losses)) / (k + 2)
    episode_mean_actor_loss.append(mean_actor_loss if not isnan(mean_actor_loss) else 0)

    mean_critic_loss = ((k + 1) * previous_mean_critic_loss + np.mean(critic_losses)) / (k + 2)
    episode_mean_critic_loss.append(mean_critic_loss if not isnan(mean_critic_loss) else 0)

    k += 1

    if print_movements:
        episodes_movements[episode] = episode_movements
    final_episode_reward = reward
    stop_time = time.time()

    episode_time = time.time() - start #+ tempo
    execution_times.append(episode_time)
    print(f'Episode {episode}/{n_episodes} '
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
                     episode_mean_rewards, episode_qualified_mean_rewards, episode_mean_actor_loss,
                     episode_mean_critic_loss, best_episode, best_step,best_reward, best_positions,
                     best_info, print_movements, episodes_movements, "w")
        print(f"fMetrics and Checkpoints saved. Episode{episode}")

print(f'For {n_episodes} episodes, there was {sum(improvements)} improvements '
      f'({round(sum(improvements) * 100 / n_episodes, 2)}%) and '
      f'{n_episodes - sum(improvements)} worse results ('
      f'{round((n_episodes - sum(improvements)) * 100 / n_episodes, 2)}%)')
# Salvar modelo final
save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, checkpoint_filename)
# Salvar métricas finais
save_metrics(results_file_name, best_positions_file_name, movements_file_name, execution_times,
             episode_mean_rewards, episode_qualified_mean_rewards, episode_mean_actor_loss,
             episode_mean_critic_loss, best_episode, best_step, best_reward, best_positions,
             best_info, print_movements, episodes_movements, "w")
print("Treinamento concluído!")
ns3_env.close()
