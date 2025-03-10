#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
 echo "Uso: $0 -n <n> -d <DV> -e <EP> -s <ST> -v <VERBOSE>"
 echo "  -n <n>    Valor para 'n' (ex: 8)"
 echo "  -d <DV>   Valor para 'DV' (ex: 50)"
 echo "  -e <EP>   Valor para 'EP' (ex: 100)"
 echo "  -s <ST>   Valor para 'ST' (ex: 100)"
 echo "  -g <GW>   Valor para 'GW' (ex: 4)"
 echo "  -v <VERBOSE> Habilita modo VERBOSE (opcional)"
 exit 1
}

# Inicializa VERBOSE como falso
VERBOSE=0

# Processa os argumentos de entrada
while getopts ":n:d:e:s:g:v" opt; do
 case $opt in
   n) VP=${OPTARG} ;;     # Valor para n
   d) DV=${OPTARG} ;;    # Valor para DV
   e) EP=${OPTARG} ;;    # Valor para EP
   s) ST=${OPTARG} ;;    # Valor para ST
   g) GW=${OPTARG} ;;    # Valor para ST
   v) VERBOSE=1 ;;       # Ativa o modo VERBOSE
   *) usage ;;           # Exibe uso correto se algo inválido for passado
 esac
done

# Verifica se todos os parâmetros obrigatórios foram fornecidos
if [ -z "$VP" ] || [ -z "$DV" ] || [ -z "$EP" ] || [ -z "$ST" ] || [ -z "$GW" ]; then
 usage
fi

# Função para exibir mensagens quando VERBOSE está habilitado
log_verbose() {
 if [ "$VERBOSE" -eq 1 ]; then
   echo "[VERBOSE] $1"
 fi
}

trap "echo 'Interrupção detectada! Encerrando...' ; kill $(jobs -p) ; exit" SIGINT SIGTERM

# Loop para executar com seed de 1 a 10
for SEED in $(seq 1 6); do
#  for GW in $(seq 2 5); do
  log_verbose "Executando SEED=$SEED e GW=$GW..."
  python3 PPO_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV \
    --gw $GW --ep $EP --st $ST --ss 1 --so 1 --sd $SEED --out 0

  if [ $(jobs -r | wc -l) -ge 4 ]; then
    wait -n
  fi
#  done
done
wait

echo "Execução concluída."