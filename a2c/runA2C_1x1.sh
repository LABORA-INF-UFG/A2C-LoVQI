#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
 echo "Uso: $0 -n <n> -d <DV> -g <GW> -e <EP> -p <ST> -s <SD> [-v]"
 echo "  -n <n>    Valor para 'Nº CANDIDATE POSITIONS' (ex: 8 means 8x8-64)"
 echo "  -d <DV>   Valor para 'DEVICES' (ex: 50)"
 echo "  -g <GW>   Valor para 'GATEWAYS' (ex: 6)"
 echo "  -e <EP>   Valor para 'EPISODE' (ex: 100)"
 echo "  -p <ST>   Valor para 'STEP' (ex: 100)"
 echo "  -s <SD>   Valor para 'SEED' (ex: 1)"
 echo "  -v        Habilita modo VERBOSE (opcional)"
 exit 1
}

# Inicializa VERBOSE como falso
VERBOSE=0

# Processa os argumentos de entrada
while getopts ":n:d:g:e:p:s:v" opt; do
 case $opt in
   n) VP=${OPTARG} ;;     # Valor para n
   d) DV=${OPTARG} ;;    # Valor para DV
   g) GW=${OPTARG} ;;    # Valor para GW
   e) EP=${OPTARG} ;;    # Valor para EP
   p) ST=${OPTARG} ;;    # Valor para ST
   s) SD=${OPTARG} ;;    # Valor para ST
   v) VERBOSE=1 ;;       # Ativa o modo VERBOSE
   *) usage ;;           # Exibe uso correto se algo inválido for passado
 esac
done

# Verifica se todos os parâmetros obrigatórios foram fornecidos
if [ -z "$VP" ] || [ -z "$DV" ] || [ -z "$GW" ] || [ -z "$EP" ] || [ -z "$ST" ] || [ -z "$SD" ]; then
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
log_verbose "Executando com SEED=$SD: python3 A2C_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --sd $SD --so 1 --ss 1"
python3 A2C_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --sd $SD --so 1 --ss 1


echo "Execução concluída."