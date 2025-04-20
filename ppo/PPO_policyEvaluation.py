import argparse
import os
import torch
import numpy as np
from ns3gym import ns3env  # Ambiente NS-3
from PPO_mapping import PPOMapping, load_checkpoint

# Configuração do dispositivo
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print(f"Executando avaliação no dispositivo: {device}")


# Criar o parser de argumentos
parser = argparse.ArgumentParser(description='Avaliação da Política A2C')
parser.add_argument('--pr', type=int, help='Número da porta')
parser.add_argument('--gr', type=int, help='Dimensão da grade (gr x gr)')
parser.add_argument('--sz', type=int, help='Tamanho do lado da área')
parser.add_argument('--dv', type=int, help='Número de Dispositivos')
parser.add_argument('--gw', type=int, help='Número de Gateways')
parser.add_argument('--ep', type=int, help='Número de episódios')
parser.add_argument('--st', type=int, help='Número de passos')
parser.add_argument('--sd', type=int, help='Número da seed')
parser.add_argument('--sp', type=int, help='Número da seed da política de treinamento')
args = parser.parse_args()

# Parâmetros da simulação
port = args.pr
dim_grid = args.gr
area_side = args.sz
nDevices = args.dv
nGateways = args.gw
n_episodes = args.ep
steps_per_episode = args.st
sim_seed = args.sd
seed_policy = args.sp

simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 1,
           "--virtualPositions": dim_grid**2,
           "--startOptimal": 1,
           "--areaSide": area_side}
print(simArgs)
# Caminho do checkpoint
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"
checkpoint_filename = f"{CHECKPOINT_DIR}PPO_chkpt_{dim_grid**2}V_{nGateways}G_{nDevices}D_{seed_policy}S.pth"

# Inicializar ambiente NS-3
simArgs = {"--nDevices": nDevices, "--nGateways": nGateways, "--vgym": 1, "--verbose": 0, "--areaSide": area_side}
ns3_env = ns3env.Ns3Env(port=port, startSim=1, simSeed=sim_seed, simArgs=simArgs, debug=0, spath="/home/rogerio/git/ns-allinone-3.42/ns-3.42/")

# Inicializar agente PPO
state_size = nGateways * 3
action_space = [i for i in range(5 ** nGateways)]  # 5 movimentos possíveis por UAV
agent = PPOMapping(ns3_env, state_size, action_space, nGateways, dim_grid, device=device)

# Carregar pesos treinados
if os.path.exists(checkpoint_filename):
    load_checkpoint(agent.policy_network, agent.value_network, agent.policy_optimizer, agent.value_optimizer, checkpoint_filename)
    print("Modelo carregado com sucesso!")
else:
    raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_filename}")

# Executar avaliação
total_rewards = []
for episode in range(n_episodes):
    state = ns3_env.reset(simSeed=sim_seed)
    total_reward = 0
    for step in range(steps_per_episode):
        action, _ = agent.select_action(state)
        next_state, reward, done, _ = ns3_env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    total_rewards.append(total_reward)
    print(f"Episódio {episode + 1}/{n_episodes}, Recompensa: {total_reward}")

# Resultados
print(f"Recompensa Média: {np.mean(total_rewards):.2f}, Desvio Padrão: {np.std(total_rewards):.2f}")
ns3_env.close()
