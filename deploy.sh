#!/bin/bash

# Definir a data para o commit
DATA=$(date +"%Y-%m-%d %H:%M:%S")

# Apagar os diretórios data e json com sudo
sudo rm -rf /opt/jimsp.github.io/data
sudo rm -rf /opt/jimsp.github.io/frontend/json

# Executar o script DVIBatch.py
sudo python3 /opt/jimsp.github.io/DVIBatch.py

# Navegar até o repositório Git
cd /opt/jimsp.github.io || exit

# Adicionar mudanças ao Git
sudo git add .

# Commit com mensagem contendo a data
sudo git commit -m "Atualização automática - $DATA"

# Push para o repositório remoto
sudo git push
