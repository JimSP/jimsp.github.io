#!/bin/bash

# Definir a data para o commit
DATA=$(date +"%Y-%m-%d %H:%M:%S")

# Navegar at� o reposit�rio Git
cd /opt/jimsp.github.io

sudo git pull

# Executar o script DVIBatch.py
sudo python3 /opt/jimsp.github.io/DVIBatch.py

# Adicionar mudan�as ao Git
sudo git add .

# Commit com mensagem contendo a data
sudo git commit -m "Atualiza��o autom�tica - $DATA"

# Push para o reposit�rio remoto
sudo git push
