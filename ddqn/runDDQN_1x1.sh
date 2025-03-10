#!/bin/bash

# Parâmetros fixos
SZ=20000 # Tamanho fixo

# Função para exibir o uso correto do script
usage() {
 echo "Uso: $0 -n <n> -d <DV> -g <GW> -i <EPI> -f <EPF> -s <ST> [-v]"
 echo "  -n <n>    Valor para 'n' (ex: 8)"
 echo "  -d <DV>   Valor para 'DV' (ex: 50)"
 echo "  -g <GW>   Valor para 'GW' (ex: 6)"
 echo "  -i <EP>   Valor para 'EP' (ex: 100)"
 echo "  -f <EP>   Valor para 'EP' (ex: 100)"
 echo "  -s <ST>   Valor para 'ST' (ex: 100)"
 echo "  -v        Habilita modo VERBOSE (opcional)"
 exit 1
}

# Inicializa VERBOSE como falso
VERBOSE=0

# Processa os argumentos de entrada
while getopts ":n:d:g:i:f:s:v" opt; do
 case $opt in
   n) VP=${OPTARG} ;;     # Valor para n
   d) DV=${OPTARG} ;;    # Valor para DV
   g) GW=${OPTARG} ;;    # Valor para GW
   i) EPI=${OPTARG} ;;    # Valor para EPInicial
   f) EPF=${OPTARG} ;;    # Valor para EPFinal
   s) ST=${OPTARG} ;;    # Valor para ST
   v) VERBOSE=1 ;;       # Ativa o modo VERBOSE
   *) usage ;;           # Exibe uso correto se algo inválido for passado
 esac
done

# Verifica se todos os parâmetros obrigatórios foram fornecidos
if [ -z "$VP" ] || [ -z "$DV" ] || [ -z "$GW" ] || [ -z "$EPI" ] || [ -z "$EPF" ] || [ -z "$ST" ]; then
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

# Executa o agente DQN
log_verbose "Executando: python3 DQN_Ns3Simulation.py --v  $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --ep $EP --st $ST --ss 1 --so 1"
python3 DQN_ns3Simulation.py --v $VERBOSE --pr 0 --gr $VP --sz $SZ --dv $DV --gw $GW --epi $EPI --epf $EPF --st $ST --ss 1 --so 1

# Gera os gráficos usando os mesmos parâmetros
log_verbose "Gerando gráfico: python3 DQN_grafs.py --v $VP_SQUARE --g $GW --d $DV"
python3 DQN_grafs.py --v $VP_SQUARE --g $GW --d $DV

echo "Execução concluída."