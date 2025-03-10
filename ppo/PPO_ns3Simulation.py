import argparse
import itertools
import os
import time
from math import isnan
import numpy as np
import pandas as pd
import torch
from ns3gym import ns3env  # Ambiente NS-3
from PPO_mapping import PPOMapping, save_metrics, load_checkpoint, save_checkpoint

# Verifica se CUDA/GPU está disponível e seta device
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

# Parâmetros e configurações do PPO
port = args.pr
dim_grid = args.gr  # Tamanho da grade
area_side = args.sz  # Lado da área de simulação
nDevices = args.dv  # Número de dispositivos
nGateways = args.gw  # Número de gateways
verbose = False if args.v == 0 else True  # Verbosidade do ambiente
debug = False  # Depuração do ambiente
plot_data = True  # Gerar gráficos
start_sim = args.ss # Iniciar o ambiente ns-3 a partir do agente
start_optimal = args.so # Iniciar os drones nas posições otimizadas
debug = 0 if start_sim == 1 else 1  # Iniciar simulação em condições ótimas
episodes = args.ep  # Número de episódios
steps_per_episode = args.st  # Passos por episódio
virtual_positions = dim_grid * dim_grid
step_size = area_side / dim_grid  # Tamanho do passo
print_movements = True
sim_seed = args.sd  # Semente de simulação
# Parâmetros para o ns-3
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--virtualPositions": virtual_positions,
           "--startOptimal": start_optimal,
           "--areaSide": area_side}

movements = ['up', 'right', 'down', 'left', 'stay']

# Caminho para salvar arquivos
TRAINNING_FILES_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"
results_file_name = f"{TRAINNING_FILES_DIR}PPO_results_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
movements_file_name = f"{TRAINNING_FILES_DIR}PPO_episodes_movements_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
best_positions_file_name = f"{TRAINNING_FILES_DIR}PPO_best_positions_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.dat"
checkpoint_filename = f"{CHECKPOINT_DIR}PPO_chkpt_{virtual_positions}V_{nGateways}G_{nDevices}D_{sim_seed}S.pth"

# Define espaço de ações e estados e tamanho da área de ação dos drones
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

# Hiperparâmetros PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_BETA = 0.05
MINI_BATCH_SIZE = 128
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
WEIGHT_DECAY = 5e-5

if nGateways < 4:
    LEARNING_RATE = 2e-4  # Reduzido para evitar oscilações
else:
    LEARNING_RATE = 3e-4  # Mantém valor padrão

# Inicializar o agente de PPO
agent = PPOMapping(
    ns3_env=ns3_env,
    state_size=state_size,
    action_space=action_space,
    n_vants=nGateways,
    dim_grid=dim_grid,
    gamma=GAMMA,
    lambdaa=GAE_LAMBDA,
    clip_epsilon=CLIP_EPSILON,
    learning_rate=LEARNING_RATE,
    batch_size=MINI_BATCH_SIZE,
    memory_limit=MINI_BATCH_SIZE * 20,
    epochs=episodes,
    entropy_coef=ENTROPY_BETA,
    device=device,
    epsilon=EPSILON,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    weight_decay=WEIGHT_DECAY
)

# Armazenamento de dados
episode_mean_policy_loss = [0]
episode_mean_value_loss = [0]
episode_mean_rewards = [0]
episode_mean_q_rewards = [0]
execution_times = []
best_reward = -np.inf
# episodes_visited_states = {}
improvements = []
# map de episodes_movements
episodes_movements = {}

# Carrega os pesos da rede
if os.path.exists(checkpoint_filename):
    load_checkpoint(actor_model=agent.policy_network,
                    critic_model=agent.value_network,
                    actor_optimizer=agent.policy_optimizer,
                    critic_optimizer=agent.value_optimizer,
                    checkpoint_filename=checkpoint_filename)

allowed_intermediate_checkpoint = 50

# Se foi interrompido, carrega o último episódio executado
if os.path.exists(results_file_name):
    data = pd.read_csv(results_file_name)
    initial_episode = data['episodio'].max() + 1
    final_episode = episodes + initial_episode
else:  # Primeira execução
    initial_episode = 1
    final_episode = episodes

k = 0
kq = 0
done = False
out_of_area = 0
accumulated_out_of_area_penalty = 0

for episode in range(initial_episode, final_episode + 1):
    start_time = time.time()  # Início da medição do tempo
    # visited_states = [state]
    episode_movements = []
    rewards = []
    q_rewards = []
    step = 0
    _ = ns3_env.reset(simSeed=sim_seed)
    state, reward, _, _ = ns3_env.get_state()
    initial_episode_reward = reward

    while step < steps_per_episode:
        state, reward, _, _ = ns3_env.get_state()

        action, log_prob = agent.select_action(state)

        # Executa a ação e obtém o próximo estado e recompensa
        if action >= len(action_space) or action < 0:
            print(f"Erro ao executar select_action com Action={action}")
        else:
            try:
                next_state, reward, done, info = ns3_env.step(action)
            except Exception as e:
                print(f"Erro ao executar ns3_env.step com action:{action}: {e}")
                raise

        if verbose:
            print(f'Step:{step} Rw:{reward} Action:{action} Info:{info}') # St:{state} Next:{next_state}')

        if done:  # Em caso de colisão termina o episódio
            break
        # Tentativa de evitar múltiplas saídas de área simultâneas
        if reward == -1: # Em caso de saída da área aumenta a penalidade proporcionalmente
            out_of_area += 1
            accumulated_out_of_area_penalty = out_of_area * 0.25
        else:
            out_of_area = 0
            accumulated_out_of_area_penalty = 0

        # Registra o estado da melhor recompensa por episódio
        if reward > best_reward:
            best_reward = reward
            best_positions = next_state
            best_info = info
            best_episode = episode
            best_step = step
        agent.remember(state, action, reward, next_state, done, log_prob)
        rewards.append(reward)

        if info and "OutOfArea" not in info:
            reward += 0.15
            q_rewards.append(reward)
            q_reward = reward
        else:
            q_reward = 0
            reward -= 0.25

        if print_movements:
            str_state = str(state).replace("[ ", "").replace("\n", "").replace("\t", "").replace("    ", ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            str_next_state = str(next_state).replace("[ ", "").replace("\n", "").replace("\t", "").replace("    ",
                                                                                                           ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            episode_movements.append((step, str_state, str_next_state, reward, q_reward, info, action, step_size))
        state = next_state
        step += 1

    if "Collision" in info: # em caso de colisão salta para o próximo episódio
        continue

    episode_time = time.time() - start_time
    execution_times.append(episode_time)
    # Atualiza a rede após o episódio
    policy_loss, value_loss = agent.update(epochs=steps_per_episode)

    # Calcula as médias dos registros
    mean_reward = ((k + 1) * episode_mean_rewards[-1] + np.mean(rewards)) / (k + 2)
    episode_mean_rewards.append(mean_reward if not isnan(mean_reward) else 0)
    mean_q_reward = ((k + 1) * episode_mean_q_rewards[-1] + np.mean(q_rewards)) / (k + 2)
    episode_mean_q_rewards.append(mean_q_reward if not isnan(mean_q_reward) else 0)
    mean_policy_loss = ((k + 1) * episode_mean_policy_loss[-1] + policy_loss) / (k + 2)
    episode_mean_policy_loss.append(mean_policy_loss if not isnan(mean_policy_loss) else 0)
    mean_value_loss = ((k + 1) * episode_mean_value_loss[-1] + value_loss) / (k + 2)
    episode_mean_value_loss.append(mean_value_loss if not isnan(mean_value_loss) else 0)
    k += 1
    if print_movements:
        episodes_movements[episode] = episode_movements
    final_episode_reward = reward
    stop_time = time.time()
    improvements.append(1 if final_episode_reward > initial_episode_reward else 0)

    if verbose:
        print(f'Episode {episode}/{episodes}'
              f' Time elapsed: {stop_time - start_time}'
              f' - Total reward: {mean_reward:.4f} -'
              f' Losses: Policy:{mean_policy_loss:.4f} Value:{mean_value_loss:.4f}\n')

    # Verifica se é hora de salvar o modelo
    if episode % allowed_intermediate_checkpoint == 0 or episode == episodes:
        save_checkpoint(agent.policy_network,
                        agent.value_network,
                        agent.policy_optimizer,
                        agent.value_optimizer,
                        checkpoint_filename)
        save_metrics(results_file_name,
                     best_positions_file_name,
                     movements_file_name,
                     execution_times,
                     episode_mean_rewards,
                     episode_mean_q_rewards,
                     episode_mean_policy_loss,
                     episode_mean_value_loss,
                     best_episode,
                     best_step,
                     best_reward,
                     best_positions,
                     best_info,
                     print_movements,
                     episodes_movements,
                     file_mode="a")

print(f'For {episodes} episodes, there was {sum(improvements)} improvements '
      f'({round(sum(improvements) * 100 / episodes, 2)}%) and '
      f'{episodes - sum(improvements)} worse results ('
      f'{round((episodes - sum(improvements)) * 100 / episodes, 2)}%)')
# Salvar modelo e métricas finais
save_checkpoint(agent.policy_network,
                agent.value_network,
                agent.policy_optimizer,
                agent.value_optimizer,
                checkpoint_filename)
save_metrics(results_file_name,
             best_positions_file_name,
             movements_file_name,
             execution_times,
             episode_mean_rewards,
             episode_mean_q_rewards,
             episode_mean_policy_loss,
             episode_mean_value_loss,
             best_episode,
             best_step,
             best_reward,
             best_positions,
             best_info,
             print_movements,
             episodes_movements,
             file_mode="w")
print("Treinamento concluído!")
ns3_env.close()
