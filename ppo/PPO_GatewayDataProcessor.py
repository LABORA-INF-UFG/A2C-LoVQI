import os
import pandas as pd

# Configurações
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo"
output_file = os.path.join(input_dir, "PPO_results.dat")
vp = 144  # Número fixo de posições virtuais
dp = 100  # Número fixo de dispositivos

# Lista de gateways e quantidade de seeds a analisar (personalize conforme os dados disponíveis)
gateways = [2, 3, 4, 5]  # Exemplo de gateways usados no código fornecido
seeds = range(1,21,1)  # Exemplo de seeds (adapte para a quantidade de seeds disponíveis nos arquivos)

# Variável para armazenar todas as médias processadas
all_results = []

# Processamento dos dados para cada número de gateways
for gp in gateways:
    # Lista para armazenar os dados de todas as seeds
    combined_data = []

    for seed in seeds:
        # Caminho do arquivo de entrada da seed para o respectivo número de gateways
        input_file = os.path.join(input_dir, f"OLD/PPO_results_{vp}V_{gp}G_{dp}D_{seed}S-OLD.dat")

        if not os.path.exists(input_file):
            print(f"Arquivo não encontrado: {input_file}")
            continue

        # Lê o arquivo de entrada como DataFrame
        print(f"OLD/PPO_results_{vp}V_{gp}G_{dp}D_{seed}S-OLD.dat")
        data = pd.read_csv(input_file)

        # Remove duplicados baseados no episódio
        data = data.drop_duplicates(subset=["episodio"], keep="first")
        # Filtrar os dados para apenas os primeiros 300 episódios
        data = data[data['episodio'] <= 300]
        # Adiciona a seed aos dados para identificação
        data["seed"] = seed

        # Armazena os dados em uma lista de dados combinados
        combined_data.append(data)

    # Combina os dados de todas as seeds para o gateway atual
    if combined_data:
        combined_data = pd.concat(combined_data)

        # Calcula a média por episódio (agrupando os dados por episódio)
        media_por_episodio = combined_data.groupby("episodio").mean().reset_index()

        # Adiciona a informação do número de gateways
        media_por_episodio["gateways"] = gp

        # Armazena os resultados no conjunto final
        all_results.append(media_por_episodio)

# Combina os resultados de todos os gateways em um único DataFrame
if all_results:
    all_results = pd.concat(all_results)

    # Ordena os dados por número de gateways e episódios
    all_results = all_results.sort_values(by=["gateways", "episodio"])

    # Remove colunas desnecessárias para gerar o CSV final, como a média da coluna "seed"
    all_results = all_results.drop(columns=["seed"], errors="ignore")

    # Salva os resultados no arquivo de saída
    all_results.to_csv(output_file, index=False)
    print(f"Arquivo salvo com sucesso em: {output_file}")
else:
    print("Nenhum dado foi processado. Verifique os arquivos de entrada.")