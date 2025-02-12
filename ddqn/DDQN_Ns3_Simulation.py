import argparse
import itertools
import time

import numpy as np
import torch
from codetiming import Timer
from numpy import random
from ns3gym import ns3env  # Ambiente NS-3
from ppo.ppomapping import PPOMapping

# Temporizador para execução
execution_time = Timer(text="Execution time: {0:.2f} seconds")
execution_time.start()

# Verifica se CUDA/GPU está disponível
if not torch.cuda.is_available():
    print("CUDA não disponível, treinamento usará CPU.")
    device = torch.device("cpu")
else:
    # Configuração do dispositivo (CPU/GPU)
    device = torch.device("cuda")
    print(f"Executando treinamento no dispositivo: {device}")

# Caminho para salvar arquivos
path_files = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"

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
parser.add_argument('--out', type=int, help='Plot the results')

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
sim_seed = random.randint(1, 20) # Semente de simulação
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

agent = DoubleDQNAgent(state_size, action_size)

for episode in range(episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.policy_network(state_tensor)
            td_error = reward + agent.gamma * q_values.max().item() * (1 - done) - q_values.max().item()

        agent.remember(state, action, reward, next_state, done, abs(td_error))
        agent.train()

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

env.close()
