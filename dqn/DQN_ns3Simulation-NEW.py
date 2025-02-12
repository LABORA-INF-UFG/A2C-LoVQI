import argparse
import itertools
import os
import time

import numpy as np
import torch
from codetiming import Timer
from numpy import random
from dqnmapping import DQNMapping
from ns3gym import ns3env

execution_time = Timer(text="Execution time: {0:.2f} seconds")
execution_time.start()
if not torch.cuda.is_available():
    raise Exception('CUDA is not available. Aborting.')
else:
    print("CUDA available.")

path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"

# Configuração do dispositivo para GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Executando no dispositivo: {device}")

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
parser.add_argument('--ep', type=int, help='Number of episodes')
parser.add_argument('--st', type=int, help='Number of steps')
parser.add_argument('--ss', type=int, help='Start NS-3 Simulation')
parser.add_argument('--so', type=int, help='Start Optimal')
parser.add_argument('--out', type=int, help='Plot the results')
parser.add_argument('--out_term', type=int, help='Output type [file or screen] for results plotting')
parser.add_argument('--progress', type=int, help='Show progress bar')
args = parser.parse_args()

# Configurações principais extraídas do QL.py
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
sim_seed = random.randint(1, 20)  # Semente de simulação
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
    state_size=state_size
)

# Configuração de parâmetros do DQN
agent.epsilon = 1.0  # Exploração inicial
agent.epsilon_min = 0.1  # Valor mínimo do epsilon
agent.epsilon_decay = 0.995  # Decaimento do epsilon
agent.gamma = 0.995  # Fator de desconto
agent.batch_size = 24  # Tamanho do lote para aprendizado
agent.learning_rate = 1e-3  # Taxa de aprendizado

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
if os.path.exists(f"{path_files}dqn_policy_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt"):
    agent.policy_network.load_state_dict(
        torch.load(f"{path_files}dqn_policy_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt"))
if os.path.exists(f"{path_files}dqn_target_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt"):
    agent.target_network.load_state_dict(
        torch.load(f"{path_files}dqn_target_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt"))

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
for episode in range(episodes):
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
    for step in range(steps_per_episode):
        # O agente decide qual ação executar
        action = agent.get_action(state)
        # Pegar o indice da ação
        iAction = action_space.index(action)
        # Aplica a ação no ambiente e obtém o próximo estado, recompensa e se o episódio acabou
        next_state, reward, done, info = ns3_env.step(iAction)
        # Armazena essa transição na memória de replay
        agent.remember(state, action, reward, next_state, done)
        step_rewards.append(reward)

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
            print(f'Step: {step} Rw:{reward} {"I:" + info}')
        if done:  # Se o episódio terminou
            break  # Sai do loop interno

    # Atualiza a rede-alvo periodicamente
    if episode % 5 == 0:
        agent.update_target_network()
    final_episode_reward = reward
    stop_time = time.time()
    # Verifica se há valores em results para calcular médias seguras
    previous_mean_reward = results[1][-1] if len(results[1]) > 0 else 0
    previous_mean_loss_episode = results[2][-1] if len(results[2]) > 0 else 0

    # Calcula as médias acumuladas
    mean_reward = ((k + 1) * previous_mean_reward + np.mean(step_rewards)) / (k + 2)
    if info and "Collision" not in info and "OutOfArea" not in info:
        qualified_mean_reward = ((k + 1) * previous_mean_reward + np.mean(step_rewards)) / (k + 2)

    mean_loss_episode = ((k + 1) * previous_mean_loss_episode + np.mean(step_losses)) / (k + 2)

    # Armazena os resultados no formato adequado
    results[0].append(stop_time - start_time)
    results[1].append(mean_reward)
    results[2].append(qualified_mean_reward if qualified_mean_reward not in [np.nan, np.inf] else 0)
    results[3].append(mean_loss_episode)

    k = k + 1
    if print_movements:
        episodes_movements[episode] = episode_movements

    improv = '+' if final_episode_reward > initial_episode_reward == True else '-'
    if improv == '+':
        improvements.append(1)
    else:
        improvements.append(0)

    print(f'Episode {episode + 1}/{episodes} {improv}'
          f' Time elapsed: {stop_time - start_time}'
          f' Mean reward: {mean_reward}'
          f' Qualified Mean reward: {qualified_mean_reward}'
          f' Mean Loss: {mean_loss_episode}\n')

    # Verifica se é hora de salvar o modelo
    if episode % (episode // 5) == 0 or episode == episodes:
        checkpoint_name_policy = f"DQN_policy_chkpt_{episode}_{virtual_positions}V_{nGateways}G_{nDevices}d.pth"
        checkpoint_name_target = f"DQN_target_chkpt_{episode}{virtual_positions}V_{nGateways}G_{nDevices}d.pth"

        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        policy_path = os.path.join(CHECKPOINT_DIR, checkpoint_name_policy)
        target_path = os.path.join(CHECKPOINT_DIR, checkpoint_name_target)

        torch.save(agent.policy_network.state_dict(), policy_path)
        torch.save(agent.target_network.state_dict(), target_path)

print(
    f'For {episodes} episodes, there was {sum(improvements)} improvements ({round(sum(improvements) * 100 / episodes, 2)}%) and {episodes - sum(improvements)} worse results ({round((episodes - sum(improvements)) * 100 / episodes, 2)}%)')

# Tempos de execução por episódio
file_name = f"{path_files}DQN_results_{virtual_positions}V_{nGateways}G_{nDevices}D.dat"
with open(file_name, "w") as file:
    file.write("episodio,tempo,reward, qualified_reward,loss\n")
    for idx, (time_elapsed, reward, qualified_mean_reward, loss) in enumerate(
            zip(results[0], results[1], results[2], results[3])):
        file.write(f"{idx + 1},{time_elapsed:.4f},{reward:.4f}, {qualified_mean_reward:.4f},{loss:.4f}\n")

if print_movements:
    file_name = f"{path_files}DQN_episodes_movements_{virtual_positions}V_{nGateways}G_{nDevices}D.dat"
    with open(file_name, "w") as file:
        file.write("episodio,step,state,next_state,reward,info,action,step_size\n")
        for idx, (episode, episode_movements) in enumerate(episodes_movements.items()):
            for step, state, next_state, reward, info, action, step_size in episode_movements:
                file.write(f"{idx + 1},{step},{state},{next_state},{reward},{info},{action},{step_size}\n")

# Salvar modelo treinado e finalizar
torch.save(agent.policy_network.state_dict(),
           f"{path_files}dqn_policy_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt")
torch.save(agent.target_network.state_dict(),
           f"{path_files}dqn_target_net_{virtual_positions}V_{nGateways}G_{nDevices}d.pt")
execution_time.stop()
print("Treinamento concluído!")
ns3_env.close()
