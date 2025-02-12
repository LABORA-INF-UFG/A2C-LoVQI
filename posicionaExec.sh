#!/bin/bash

# Parâmetro com o nome do método
METHOD_NAME=$1
echo "Método: $METHOD_NAME"

# Diretório do projeto
PROJECT_DIR="$HOME/git/A2C-LoVQI"

# Verificar se o diretório do projeto existe
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Erro: Diretório do projeto não encontrado: $PROJECT_DIR"
    exit 1
fi

# Ir para o diretório do projeto
cd "$PROJECT_DIR" || exit

# Ativar o ambiente virtual
if [ -f "tf/bin/activate" ]; then
    source tf/bin/activate
else
    echo "Erro: Ambiente virtual não encontrado."
    exit 1
fi

# Normalizar método para diretório (letras minúsculas)
METHOD_DIR=$(echo "$METHOD_NAME" | tr '[:upper:]' '[:lower:]')

# Verificar se o diretório correspondente ao método existe
if [ ! -d "$METHOD_DIR" ]; then
    echo "Erro: Diretório correspondente ao método não encontrado: $METHOD_DIR"
    exit 1
fi

# Acessar o diretório correspondente
cd "$METHOD_DIR" || exit