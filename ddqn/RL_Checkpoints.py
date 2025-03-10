import torch

def load_checkpoint(p_model, v_model, optimizer, filename):
    """Carrega o estado do modelo e otimizadores de um arquivo."""
    checkpoint = torch.load(filename)

    p_model.load_state_dict(checkpoint['policy_model_state_dict'])
    v_model.load_state_dict(checkpoint['value_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Carga da rede neural {filename} realizada com sucesso!")

def save_checkpoint(p_model, v_model, optimizer, filename):
    """Salva o estado atual do modelo e otimizadores."""
    checkpoint = {
        'policy_model_state_dict': p_model.state_dict(),
        'value_model_state_dict': v_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint da rede neural salvo em {filename}")

def save_simulation_results(
        results_file_name,
        movements_file_name,
        best_positions_file_name,
        results,
        initial_episode,
        episodes_movements,
        best_episode,
        best_step,
        best_reward,
        best_positions,
        best_info,
        print_movements=False,
        file_mode="w"
):
    """
    Função para salvar diversos resultados de simulação em arquivos.

    Args:
        results_file_name (str): Nome do arquivo para salvar resultados gerais.
        movements_file_name (str): Nome do arquivo para salvar movimentos dos episódios.
        best_positions_file_name (str): Nome do arquivo para salvar os melhores resultados.
        results (list): Resultados gerais contendo [tempos, recompensas, recompensas qualificadas, perdas].
        initial_episode (int): Número do episódio inicial.
        episodes_movements (dict): Movimentos registrados por episódio (estado, próximo estado, etc.).
        best_episode (int): Número do melhor episódio.
        best_step (int): Passo no melhor episódio.
        best_reward (float): Melhor recompensa alcançada.
        best_positions (any): Posição (ou configuração) associada à melhor recompensa.
        best_info (any): Informações adicionais relacionadas à melhor recompensa.
        print_movements (bool): Indica se deve salvar os movimentos em arquivo. Default é False.

    """
    import os

    # Salvando os resultados gerais
    with open(results_file_name, file_mode) as file:
        if file_mode == "w":
            file.write("episodio,tempo,reward,qualified_reward,loss\n")
        for idx, (time_elapsed, reward, qualified_mean_reward, loss) in enumerate(
                zip(results[0], results[1], results[2], results[3])):
            file.write(
                f"{idx + initial_episode},{time_elapsed:.4f},{reward:.4f},{qualified_mean_reward:.4f},{loss:.4f}\n")

    # Salvando os movimentos dos episódios (se necessário)
    if print_movements:
        with open(movements_file_name, file_mode) as file:
            if file_mode == "w":
                file.write("episodio,step,state,next_state,reward,info,action,step_size\n")
            for idx, (episode, episode_movements) in enumerate(episodes_movements.items()):
                for step, state, next_state, reward, info, action, step_size in episode_movements:
                    file.write(
                        f"{idx + initial_episode},{step},{state},{next_state},{reward},{info},{action},{step_size}\n")

    # Salvando os melhores resultados
    with open(best_positions_file_name, file_mode) as file:
        if file_mode == "w":
            file.write("episode,step,reward,positions,info\n")
        file.write(f"{best_episode},{best_step},{best_reward},{best_positions},{best_info}\n")

    print(f"Checkpoint de resultados salvos em {results_file_name}")
