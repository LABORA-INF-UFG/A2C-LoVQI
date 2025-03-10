import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
from numpy import random

from DQN_mapping import DQNMapping, load_checkpoint, save_checkpoint, save_simulation_results
from ns3gym import ns3env

device = torch.device("cpu")
if torch.cuda.is_available():
    print('CUDA is available. Using GPU.')
    # Configuração do dispositivo para GPU ou CPU
    device = torch.device("cuda")
else:
    print('CUDA is not available. Using CPU.')

# Create a parser
parser = argparse.ArgumentParser(description='DQN for n UAVs')
# Add arguments to the parser
# Comand line arguments: --v 1 --pr 0 --gr 16 --sz 20000 --dv 100 --gw 4 --ns 1 --ep 10 --st 10 --ss 1

parser.add_argument('--v', type=int, help='Verbose mode')
parser.add_argument('--pr', type=int, help='Port number')
parser.add_argument('--gr', type=int, help='Grid dimension (gr x gr)')
parser.add_argument('--sz', type=int, help='Area side size')
parser.add_argument('--dv', type=int, help='Number of Devices')
parser.add_argument('--gw', type=int, help='Number of Gateways')
parser.add_argument('--epi', type=int, help='Number of initial episode')
parser.add_argument('--epf', type=int, help='Number of final episode')
parser.add_argument('--st', type=int, help='Number of steps')
parser.add_argument('--ss', type=int, help='Start NS-3 Simulation')
parser.add_argument('--so', type=int, help='Start Optimal')
parser.add_argument('--out', type=int, help='Plot the results')
parser.add_argument('--out_term', type=int, help='Output type [file or screen] for results plotting')
parser.add_argument('--progress', type=int, help='Show progress bar')
args = parser.parse_args()

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
# episodes = args.ep  # Número de episódios
steps_per_episode = args.st  # Passos por episódio
virtual_positions = dim_grid * dim_grid
step_size = area_side / dim_grid  # Tamanho do passo
print_movements = True
sim_seed = random.randint(1, 20)  # Semente de simulação
initial_episode = args.epi
final_episode = args.epf
# run_seed = final_episode-initial_episode
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--virtualPositions": virtual_positions,
           "--startOptimal": start_optimal,
           "--areaSide": area_side}

path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"
checkpoint_filename = f"{CHECKPOINT_DIR}DQN_policy_chkpt_{virtual_positions}V_{nGateways}G_{nDevices}DD.pth"
results_file_name = f"{path_files}DQN_results_{virtual_positions}V_{nGateways}G_{nDevices}DD.dat"
best_positions_file_name = f"{path_files}DQN_best_positions_{virtual_positions}V_{nGateways}G_{nDevices}DD.dat"
movements_file_name = f"{path_files}DQN_movements_{virtual_positions}V_{nGateways}G_{nDevices}DD.dat"

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

movements = ['up', 'right', 'down', 'left', 'stay']
# Inicialização do agente DQN
# Action space é dado pelo Número de ações e na quantidade de drones
action_space = list(itertools.product(movements, repeat=nGateways))
area = dim_grid * dim_grid
state_size = nGateways * 3

agent = DQNMapping(
    ns3_env=None,  # Ambientes NS-3
    action_space=action_space,  # Espaço de ações possíveis
    dim_grid=dim_grid,  # Configuração do tamanho da grade
    n_vants=nGateways,
    state_size=state_size,
    device=device
)

# Configuração de parâmetros do DQN
agent.epsilon = 1.0  # Exploração inicial
agent.epsilon_min = 0.05  # Valor mínimo do epsilon
agent.epsilon_decay = 0.9975  # Decaimento do epsilon
agent.gamma = 0.999  # Fator de desconto
agent.batch_size = 64  # Tamanho do lote para aprendizado
agent.learning_rate = 2e-4  # Taxa de aprendizado

# Certifique-se de que `agent` utiliza o dispositivo correto (GPU ou CPU)
agent.policy_network.to(device)
agent.target_network.to(device)

# Inicialização do ambiente NS-3
ns3_env = ns3env.Ns3Env(
    port=port,
    startSim=start_sim,
    simSeed=sim_seed,
    simArgs=simArgs,
    debug=debug,
    spath="/home/rogerio/git/ns-allinone-3.42/ns-3.42/"
)

# Carregar os pesos salvos se existirem os arquivos
if os.path.exists(checkpoint_filename):
    load_checkpoint(agent.policy_network, agent.target_network, agent.optimizer, checkpoint_filename)

# Lista para registrar os dados de cada episódio
results = [[], [], [], []]
improvements = []
# Loop principal de aprendizado
done = False
# map de episodes_movements
episodes_movements = {}
episode_times = []

k = 0
qualified_mean_reward = 0

# Se foi interrompido, carrega o último episódio executado
# if os.path.exists(results_file_name):
#     data = pd.read_csv(results_file_name)
#     initial_episode = data['episodio'].max() + 1
#     final_episode = episodes + initial_episode
# else:
# Primeira execução
allowed_checkpoint = int(0.2 * final_episode - initial_episode + 1)
allowed_checkpoint = 10 if allowed_checkpoint < 10 else allowed_checkpoint
best_reward = -np.inf

for episode in range(initial_episode, final_episode + 1):
    start_time = time.time()  # Início da medição do tempo
    episode_movements = []

    agent.reset_epsilon()  # Reinicia o epsilon, se necessário
    if done:  # Colision in last episode - Reset()
        sim_seed = random.randint(1, 20)
        ns3_env.reset(sim_seed)
    else:
        state, reward, _, _ = ns3_env.get_state()
    initial_episode_reward = reward
    step_rewards = []
    step_losses = []
    step_q_rewards = []
    for step in range(steps_per_episode):
        # O agente decide qual ação executar
        action = agent.get_action(state)
        # Pegar o índice da ação
        iAction = action_space.index(action)
        # Aplica a ação no ambiente e obtém o próximo estado, recompensa, informações e se o episódio acabou
        next_state, reward, done, info = ns3_env.step(iAction)
        # Armazena essa transição na memória de replay
        agent.remember(state, action, reward, next_state, done)
        step_rewards.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_positions = next_state
            best_step = step
            best_episode = episode
            best_info = info
        if info and "Collision" not in info and "OutOfArea" not in info:
            step_q_rewards.append(reward)
        loss = agent.get_loss(next_state, reward, action)
        step_losses.append(loss)

        if print_movements:
            str_state = str(state).replace("[ ", "[").replace("\n", "").replace("\t", "").replace("    ", ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            str_next_state = str(next_state).replace("[ ", "[").replace("\n", "").replace("\t", "").replace("    ",
                                                                                                            ";").replace(
                "   ", ";").replace("  ", ";").replace(" ", ";")
            episode_movements.append((step, str_state, str_next_state, reward, info, action, step_size))

        state = next_state  # Move para o próximo estado
        if verbose:
            print(f'Episode: {episode}/{final_episode} Step: {step} Rw:{reward} {"I:" + info}')
        else:
            # a cada step imprimir \, |, /, -, \, |, /, - no mesmo lugar
            print('\r' + f'Episode: {episode + 1 if step != 0 else episode} / {final_episode}'
                         f' [{int((step / steps_per_episode) * 100) if step != 0 else 100}]%'
                         f' Time elapsed: {time.time() - start_time:.2f}s',
                  flush=True,
                  end='' if step != 0 else '\n')
        if done:  # Se o episódio terminou
            break  # Sai do loop interno

    # Atualiza a rede-alvo periodicamente
    if episode % 5 == 0:
        agent.update_target_network()
    final_episode_reward = reward
    stop_time = time.time()
    # Verifica se há valores em results para calcular médias seguras
    previous_mean_reward = results[1][-1] if len(results[1]) > 0 else 0
    previous_quali_mean_reward = results[2][-1] if len(results[2]) > 0 else 0
    previous_mean_loss_episode = results[3][-1] if len(results[3]) > 0 else 0

    # Calcula as médias acumuladas
    mean_reward = ((k + 1) * previous_mean_reward + np.mean(step_rewards)) / (k + 2)
    if info and "Collision" not in info and "OutOfArea" not in info:
        qualified_mean_reward = ((k + 1) * previous_quali_mean_reward + np.mean(step_q_rewards)) / (k + 2)
    mean_loss_episode = ((k + 1) * previous_mean_loss_episode + np.mean(step_losses)) / (k + 2)

    # Armazena os resultados no formato adequado
    results[0].append(stop_time - start_time)
    results[1].append(mean_reward)
    results[2].append(qualified_mean_reward if qualified_mean_reward not in [np.nan, np.inf] else 0)
    results[3].append(mean_loss_episode)

    k = k + 1
    if print_movements:
        episodes_movements[episode] = episode_movements

    improvements.append(1 if ((final_episode_reward > initial_episode_reward) == True) else 0)

    time_elapsed = stop_time - start_time
    if verbose:
        print(f'Episode {episode}/{final_episode} '
              f'[{'+' if ((final_episode_reward > initial_episode_reward) == True) else '-'}'
              f'] Time elapsed: {time_elapsed}'
              f' Mean reward: {mean_reward:.4f}'
              f' Qualified Mean reward: {qualified_mean_reward:.4f}'
              f' Mean Loss: {mean_loss_episode:.4f}\n')

    # Verifica se é hora de salvar o modelo
    if episode % allowed_checkpoint == 0:
        save_checkpoint(
            agent.policy_network,
            agent.target_network,
            agent.optimizer,
            f"{checkpoint_filename}")
        save_simulation_results(
            results_file_name=results_file_name,
            movements_file_name=movements_file_name,
            best_positions_file_name=best_positions_file_name,
            results=results,
            initial_episode=initial_episode,
            episodes_movements=episodes_movements,
            best_episode=best_episode,
            best_step=best_step,
            best_reward=best_reward,
            best_positions=best_positions,
            best_info=best_info,
            print_movements=print_movements,
            file_mode="w"
        )

if verbose:
    print(f'For {final_episode - initial_episode + 1} episodes, there was {sum(improvements)} '
          f'improvements ({round(sum(improvements) * 100 / final_episode - initial_episode + 1, 2)}%) '
          f'and {final_episode - initial_episode + 1 - sum(improvements)} '
          f'worse results ({round((final_episode - initial_episode + 1 - sum(improvements)) * 100 / final_episode - initial_episode + 1, 2)}%)')

save_checkpoint(
    agent.policy_network,
    agent.target_network,
    agent.optimizer,
    f"{checkpoint_filename}")
save_simulation_results(
    results_file_name=results_file_name,
    movements_file_name=movements_file_name,
    best_positions_file_name=best_positions_file_name,
    results=results,
    initial_episode=initial_episode,
    episodes_movements=episodes_movements,
    best_episode=best_episode,
    best_step=best_step,
    best_reward=best_reward,
    best_positions=best_positions,
    best_info=best_info,
    print_movements=print_movements,
    file_mode="a"
)

print("Treinamento concluído!")
ns3_env.close()
