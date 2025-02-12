#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
  echo "Uso: $0 -n <n> -d <DV> -g <GW> -e <EP> -s <ST>"
  echo "  -n <n>    Valor para 'n' (ex: 8)"
  echo "  -d <DV>   Valor para 'DV' (ex: 50)"
  echo "  -g <GW>   Valor para 'GW' (ex: 6)"
  echo "  -e <EP>   Valor para 'EP' (ex: 100)"
  echo "  -s <ST>   Valor para 'ST' (ex: 100)"
  exit 1
}

# Processa os argumentos de entrada
while getopts ":n:d:g:e:s:" opt; do
  case $opt in
    n) VP=${OPTARG} ;;     # Valor para n
    d) DV=${OPTARG} ;;    # Valor para DV
    g) GW=${OPTARG} ;;    # Valor para GW
    e) EP=${OPTARG} ;;    # Valor para EP
    s) ST=${OPTARG} ;;    # Valor para ST
    *) usage ;;           # Exibe uso correto se algo inválido for passado
  esac
done
echo $0 $1
# Verifica se todos os parâmetros foram fornecidos
if [ -z "$VP" ] || [ -z "$DV" ] || [ -z "$GW" ] || [ -z "$EP" ] || [ -z "$ST" ]; then
  usage
fi
VP_SQUARE=$(($VP * $VP))

# Executa o agente DQN
echo "Executando: python3 A2C_Ns3Simulation.py --v 1 --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --ss 1 --so 1"
python3 A2C_Ns3Simulation.py --v 1 --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --ss 1 --so 1

# Gera os gráficos usando os mesmos parâmetros
echo "Gerando gráfico: python3 A2C_grafs.py --v $VP_SQUARE --g $GW --d $DV"
python3 A2C_grafs.py --v $VP_SQUARE --g $GW --d $DV

echo "Execução concluída."