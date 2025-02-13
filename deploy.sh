#!/bin/bash

# Definir a data para o commit
DATA=$(date +"%Y-%m-%d %H:%M:%S")

# Navegar até o repositório Git
cd /opt/jimsp.github.io

sudo git pull

# Executar o script DVIBatch.py
sudo python3 /opt/jimsp.github.io/DVIBatch.py

# Adicionar mudanças ao Git
sudo git add .

# Commit com mensagem contendo a data
sudo git commit -m "Atualização automática - $DATA"

# Push para o repositório remoto
sudo git push
