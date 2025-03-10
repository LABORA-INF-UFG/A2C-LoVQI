import torch
import numpy as np
from typing import Tuple


def simulate_deployment(posvirt: int, gw: int, dev: int):
    """
    Implanta drones utilizando uma política DQN treinada.

    :param posvirt: Número de posições virtuais utilizadas no treinamento.
    :param gw: Número de gateways.
    :param dev: Número de dispositivos.
    :return: Melhor configuração para os gateways e a melhor QoS encontrada.
    """
    # Carregar o caminho do modelo treinado
    model_path = f"/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/dqn_policy_net_{posvirt}V_{gw}G_{dev}d.pt"

    # Verificar se o arquivo existe e carregar o modelo treinado
    try:
        policy_net = torch.load(model_path)
        policy_net.eval()  # Configurar o modelo para modo de avaliação (inferência)
    except FileNotFoundError:
        print(f"Modelo treinado não encontrado em {model_path}")
        return

    # Configurações do ambiente de simulação
    area_side = 1000  # Tamanho lateral da área em metros (1000 x 1000)
    step_size = 10  # Tamanho de passo da simulação, em metros
    np.random.seed(42)  # Ajustar seed para reprodutibilidade

    # Gerar posições aleatórias iniciais para os gateways
    gateways = np.random.rand(gw, 2) * area_side  # Coordenadas aleatórias no plano
    print(f"Posições iniciais aleatórias para os Gateways: {gateways}")

    # Gerar posições virtuais possíveis em que os gateways podem ser movidos
    virtual_positions = np.random.rand(posvirt, 2) * area_side
    print(f"Posições virtuais possíveis: {virtual_positions}")

    # Gerar dispositivos aleatoriamente distribuídos
    devices = np.random.rand(dev, 2) * area_side
    print(f"Dispositivos: {devices}")

    # Função para calcular QoS como a distância média entre dispositivos e gateways
    def calculate_qos(devices: np.ndarray, gateways: np.ndarray) -> float:
        """
        Calcula a métrica de QoS baseada na média das distâncias entre os dispositivos e os gateways.
        """
        distances = np.sqrt(((devices[:, None, :] - gateways[None, :, :]) ** 2).sum(axis=2))
        return -np.mean(np.min(distances, axis=1))  # Negativo para ser minimização de custo

    # Variáveis para rastrear a melhor QoS e configuração
    best_qos = float('-inf')
    best_positions = None

    # Loop por todas as posições virtuais
    for pos in virtual_positions:
        # Calcular o estado (concatenar dispositivos e posição virtual)
        state = np.hstack([devices.ravel(), pos.ravel()])

        # Prever a ação a partir da rede treinada
        with torch.no_grad():
            q_values = policy_net(torch.tensor(state, dtype=torch.float32))
            best_action_idx = torch.argmax(q_values).item()

        # Atualizar posições dos gateways de acordo com a ação escolhida
        gateway_to_move = best_action_idx % gw
        gateways[gateway_to_move] = pos

        # Avaliar a métrica de QoS para a configuração atual
        qos = calculate_qos(devices, gateways)

        # Atualizar melhores configurações, caso necessário
        if qos > best_qos:
            best_qos = qos
            best_positions = gateways.copy()

    # Resultado final
    print(f"Melhor configuração de gateways: {best_positions}")
    print(f"Melhor QoS obtida: {best_qos}")
    return best_positions, best_qos


# Função principal para execução
if __name__ == "__main__":
    # Parâmetros de entrada
    posvirt = 50  # Número de posições virtuais usadas no treinamento
    gw = 5  # Número de gateways
    dev = 100  # Número de dispositivos

    best_positions, best_qos = simulate_deployment(posvirt, gw, dev)

    if best_positions is not None:
        print("\nConfiguração Final:")
        for i, pos in enumerate(best_positions):
            print(f"Gateway {i + 1}: {pos}")
        print(f"Melhor QoS Final: {best_qos}")