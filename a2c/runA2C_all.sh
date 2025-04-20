#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
 echo "Uso: $0 -d <DV> -e <EP> -s <ST> -g <GW> -v <VERBOSE>"
 echo "  -e <EP>   Valor para 'EP' (ex: 100)"
 echo "  -s <ST>   Valor para 'ST' (ex: 100)"
 echo "  -g <GW>   Valor para 'GW' (ex: 4)"
 echo "  -v <VERBOSE> Habilita modo VERBOSE (opcional)"
 exit 1
}

# Inicializa VERBOSE como falso
VERBOSE=0

# Processa os argumentos de entrada
while getopts ":e:s:g:v" opt; do
 case $opt in
   e) EP=${OPTARG} ;;    # Valor para EP
   s) ST=${OPTARG} ;;    # Valor para ST
   g) GW=${OPTARG} ;;    # Valor para GW
   v) VERBOSE=1 ;;       # Ativa o modo VERBOSE
   *) usage ;;           # Exibe uso correto se algo inválido for passado
 esac
done

# Verifica se todos os parâmetros obrigatórios foram fornecidos
if [ -z "$GW" ] || [ -z "$EP" ] || [ -z "$ST" ]; then
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
for SEED in $(seq 1 4); do
  for VP in $(seq 7 12); do
    for DV in 50 100 150 200; do
      log_verbose "Executando SEED=$SEED, P.Cand:$VP, Devs:$DV e GW=$GW..."
      python3 A2C_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV \
        --gw $GW --ep $EP --st $ST --ss 1 --so 1 --sd $SEED --out 0 &

      if [ $(jobs -r | wc -l) -ge 3 ]; then
        wait -n
      fi
    done
  done
  wait
done

echo "Execução concluída."