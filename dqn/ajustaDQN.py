import argparse
import os
import pandas as pd


input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"

# Parser de argumentos para entrada de parâmetros
parser = argparse.ArgumentParser(description='PLOTS of DQN for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, help='Number of Gateways')
parser.add_argument('--d', type=int, help='Number of Devices')
args = parser.parse_args()

vp = args.v
gp = args.g  # Lista de gateways
dp = args.d

input_file = os.path.join(input_dir, f'DQN_results_{vp}V_{gp}G_{dp}DD.dat')
output_file = os.path.join(input_dir, f'DQN_results_{vp}V_{gp}G_{dp}DDD.dat')

# Process the input file and adjust the episode values
if os.path.exists(input_file):
    # Read the CSV file into a Pandas DataFrame
    data = pd.read_csv(input_file)

    # Adjust the 'episodio' column by subtracting 120
    data['episodio'] = data['episodio'] - 120

    # Define the output file path
    output_file = os.path.join(input_dir, f"DQN_results_{vp}V_{gp}G_{dp}DD.csv")
    # Write the adjusted data to the output file
    data.to_csv(output_file, index=False)
    print(f"Ajuste concluído: arquivo salvo em {output_file}")
else:
    print(f"Erro: o arquivo de entrada {input_file} não foi encontrado.")