#!/bin/bash

# Parâmetros fixos
EP=200   # Número de episódios fixos
ST=200   # Parâmetro de steps fixo
SZ=20000 # Tamanho fixo

# Laços para gerar as combinações de parâmetros
for n in 8 7; do
  # Calcula o número de posições virtuais (vp = n^2)
  VP=$((n * n))

  for DV in 50 100 150; do
    for GW in 6 5 4; do
      # Executa o agente DQN
      echo "Executando: python3 DQN_Ns3Simulation.py --v 0 --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --ss 1"
      python3 DQN_Ns3Simulation.py --v 0 --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --ss 1

      # Gera os gráficos usando os mesmos parâmetros
      echo "Gerando gráfico: python3 DQN_grafs.py --v $VP --g $GW --d $DV"
      python3 DQN_grafs.py --v $VP --g $GW --d $DV
    done
  done
done

echo "Execuções concluídas."