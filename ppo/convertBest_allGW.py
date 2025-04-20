import os


def process_positions(v, g, d, s):
    # Caminhos para os arquivos                                                                                 DQN_best_positions_81V_2G_50D.dat
    input_file = f"/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo/PPO_best_positions_{v}V_{g}G_{d}D_{s}S.dat"
    output_file = f"/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/opt/ppo/ppo_allGwPos_{v}V_{g}G_{d}D_{s}S.dat"

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Arquivo de entrada não encontrado: {input_file}")
        return

    try:
        # Leitura do arquivo de entrada
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Obtém a linha de dados (a segunda linha no caso)
        data_line = lines[1].strip()

        # Divide os campos separados por vírgula
        _, _, _, positions_field, _ = data_line.split(",", 4)

        # Remove os colchetes ao redor do campo positions e divide os valores
        positions = positions_field.strip("[]").split()

        # Verifica se o número de coordenadas é correto
        if len(positions) != int(g) * 3:
            print(f"Número incorreto de coordenadas no arquivo para g={g}")
            return

        # Agrupa os dados de positions em tripletas (x, y, z)
        coordinates = [positions[i:i + 3] for i in range(0, len(positions), 3)]

        # Escrita do arquivo de saída
        with open(output_file, 'w') as f:
            # Cabeçalho
            f.write("id,x,y,z\n")
            # Escrevendo cada coordenada no formato especificado
            for coord in coordinates:
                f.write(','.join(coord) + '\n')

        print(f"Arquivo processado com sucesso. Saída: {output_file}")

    except Exception as e:
        print(f"Erro ao processar o arquivo: {str(e)}")


# Exemplo de chamada da função
# Informe os valores para v, g, d e s como parâmetros
for v in [49, 64, 81, 100, 121, 144]:
    for d in [50, 100, 150, 200]:
        for g in [2, 4]:
            for s in range(2, 5):
                process_positions(v, g, d, s)