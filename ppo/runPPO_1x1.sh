#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
 echo "Uso: $0 -n <n> -d <DV> -g <GW> -e <EP> -s <ST> -r <SD> [-v]"
 echo "  -n <n>    Valor para 'n' (ex: 8)"
 echo "  -d <DV>   Valor para 'DV' (ex: 50)"
 echo "  -g <GW>   Valor para 'GW' (ex: 6)"
 echo "  -e <EP>   Valor para 'EP' (ex: 100)"
 echo "  -s <ST>   Valor para 'ST' (ex: 100)"
 echo "  -v        Habilita modo VERBOSE (opcional)"
 exit 1
}

# Inicializa VERBOSE como falso
VERBOSE=0

# Processa os argumentos de entrada
while getopts ":n:d:g:e:s:r:v" opt; do
 case $opt in
   n) VP=${OPTARG} ;;     # Valor para n
   d) DV=${OPTARG} ;;    # Valor para DV
   g) GW=${OPTARG} ;;    # Valor para GW
   e) EP=${OPTARG} ;;    # Valor para EP
   s) ST=${OPTARG} ;;    # Valor para ST
   r) SD=${OPTARG} ;;    # Valor para SD
   v) VERBOSE=1 ;;       # Ativa o modo VERBOSE
   *) usage ;;           # Exibe uso correto se algo inválido for passado
 esac
done

# Verifica se todos os parâmetros obrigatórios foram fornecidos
if [ -z "$VP" ] || [ -z "$DV" ] || [ -z "$GW" ] || [ -z "$EP" ] || [ -z "$SD" ] || [ -z "$ST" ]; then
 usage
fi

# Calcula VP_SQUARE
VP_SQUARE=$(($VP * $VP))

# Função para exibir mensagens quando VERBOSE está habilitado
log_verbose() {
 if [ "$VERBOSE" = true ]; then
   echo "[VERBOSE] $1"
 fi
}

# Loop para executar com seed de 1 a 20
log_verbose "Executando com SEED=$SEED: python3 PPO_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --sd $SD --so 1 --ss 1"
python3 PPO_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --sd $SD --so 1 --ss 1

log_verbose "Gerando gráfico para SEED=$SEED: python3 A2C_grafs.py --v $VP_SQUARE --g $GW --d $DV --s $SD"
python3 PPO_grafs.py --v $VP_SQUARE --g $GW --d $DV --s $SD

echo "Execução concluída."