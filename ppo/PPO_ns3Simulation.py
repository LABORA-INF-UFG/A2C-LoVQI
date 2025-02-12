import argparse
import itertools
import os
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
CHECKPOINT_DIR = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/checkpoints/"

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

agent = PPOMapping(
    ns3_env=ns3_env,
    state_size=state_size,
    action_space=action_space,
    n_vants=nGateways,
    dim_grid=dim_grid,
    gamma=0.99,
    lambdaa=0.95,
    clip_epsilon=0.2,
    learning_rate=0.0001,
    batch_size=64,
    epochs=episodes,
    entropy_coef=0.01,
    device='cpu'
)


# Armazenamento de dados
episode_mean_actor_loss = [0]
episode_mean_critic_loss = [0]
episode_mean_rewards = [0]
execution_times = []
episodes_visited_states = {}
improvements = []
# map de episodes_movements
episodes_movements = {}

k = 0
done = False

for ep in range(episodes):
    print(f'Episode {ep}')
    start_time = time.time()  # Início da medição do tempo
    start = time.time()
    state = ns3_env.reset()
    visited_states = [state]
    episode_movements = []
    rewards = []
    actor_losses = []
    critic_losses = []
    step = 0
    if done:  # Colision in last episode - Reset()
        sim_seed = random.randint(1, 20)
        obs = ns3_env.reset(sim_seed)
    else:
        state, reward, _, _ = ns3_env.get_state()
    initial_episode_reward = reward

    while step < steps_per_episode:
        if step % 100 == 0:
            print(f'Step {step}')

        # Converte o estado para tensor e move para o dispositivo
        state, _, _, _ = ns3_env.get_state()
        action, log_prob = agent.select_action(state)

        # Executa a ação e obtém o próximo estado e recompensa
        if action >= len(action_space) or action < 0:
            print(f"Erro ao executar choose_action com iAction={action}: iAction >= len(action_space)")
        else:
            try:
                next_state, reward, done, info = ns3_env.step(action)
            except Exception as e:
                print(f"Erro ao executar ns3_env.step com action:{action}: {e}")
                raise
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
                f'Step: {step} Rw:{reward} {"I:" + info if "Collision" not in info and "OutOfArea" not in info else ""}')  # Ac:{action} St:{next_state}')

        agent.remember(state, action, reward, next_state, done, log_prob)
        state = next_state
        rewards.append(reward)
        step += 1
    end = time.time()
    episode_time = end - start
    execution_times.append(episode_time)
    episodes_visited_states[ep] = visited_states
    # Atualiza a rede após o episódio
    agent.update()
    # Calcula as médias dos registros
    mean_reward = ((k + 1) * episode_mean_rewards[-1] + np.mean(rewards)) / (k + 2)
    episode_mean_rewards.append(mean_reward)
    mean_actor_loss = ((k + 1) * episode_mean_actor_loss[-1] + np.mean(actor_losses)) / (k + 2)
    episode_mean_actor_loss.append(mean_actor_loss)
    mean_critic_loss = ((k + 1) * episode_mean_critic_loss[-1] + np.mean(critic_losses)) / (k + 2)
    episode_mean_critic_loss.append(mean_critic_loss)
    k += 1
    if print_movements:
        episodes_movements[ep] = episode_movements
    final_episode_reward = reward
    stop_time = time.time()
    if final_episode_reward > initial_episode_reward:
        print(f'Ep: {ep} +')
        improvements.append(1)
    else:
        print(f'Ep: {ep} -')
        improvements.append(0)

    if verbose:
        print(f'Episode {ep + 1}/{episodes}'
              f' Time elapsed: {stop_time - start_time}'
              f' - Total reward: {mean_reward} -'
              f' Losses: Actor:{mean_actor_loss} Critic:{mean_critic_loss}\n')

    # Verifica se é hora de salvar o modelo
    if ep % (ep // 5) == 0 or ep == episodes:
        checkpoint_name_policy = f"PPO_policy_chkpt_{ep}_{virtual_positions}V_{nGateways}G_{nDevices}d.pth"
        checkpoint_name_value = f"PPO_value_chkpt_{ep}{virtual_positions}V_{nGateways}G_{nDevices}d.pth"
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        policy_path = os.path.join(CHECKPOINT_DIR, checkpoint_name_policy)
        value_path = os.path.join(CHECKPOINT_DIR, checkpoint_name_value)
        torch.save(agent.policy_network.state_dict(), policy_path)
        torch.save(agent.value_network.state_dict(), value_path)

print(f'For {episodes} episodes, there was {sum(improvements)} improvements '
      f'({round(sum(improvements) * 100 / episodes, 2)}%) and '
      f'{episodes - sum(improvements)} worse results ('
      f'{round((episodes - sum(improvements)) * 100 / episodes, 2)}%)')

# Tempos de execução por episódio
file_name = f"{path_files}PPO_results_{virtual_positions}V_{nGateways}G_{nDevices}D.dat"
with open(file_name, "w") as file:
    file.write("episodio,tempo,reward,actor_loss,critic_loss\n")
    for idx, (time_elapsed, reward, actor_loss, critic_loss) in enumerate(zip(execution_times, episode_mean_rewards, episode_mean_actor_loss, episode_mean_critic_loss)):
        file.write(f"{idx + 1},{time_elapsed:.4f},{reward:.4f},{actor_loss:.4f}, {critic_loss:.4f}\n")

if print_movements:
    file_name = f"{path_files}PPO_episodes_movements_{virtual_positions}V_{nGateways}G_{nDevices}D.dat"
    with open(file_name, "w") as file:
        file.write("episodio,step,state,next_state,reward,info,action,step_size\n")
        for idx, (episode, episode_movements) in enumerate(episodes_movements.items()):
            for step, state, next_state, reward, info, action, step_size in episode_movements:
                file.write(f"{idx + 1},{step},{state},{next_state},{reward},{info},{action},{step_size}\n")

execution_time.stop()
print("Treinamento concluído!")
ns3_env.close()
