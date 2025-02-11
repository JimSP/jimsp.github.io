// script.js

document.addEventListener("DOMContentLoaded", () => {
  loadAllReports();
  setupModal();
});

function loadAllReports() {
  const dash = document.getElementById("dashboard");
  dash.innerHTML = '<div class="loading">?? Carregando relatórios...</div>';

  fetch("frontend/json/main.json")
    .then(r => r.json())
    .then(data => {
      const paths = listAllFiles(data.reports);
      dash.innerHTML = ""; 

      if (paths.length === 0) {
        dash.textContent = "Nenhum relatório encontrado.";
        return;
      }

      paths.forEach(filePath => {
        fetch("frontend/json/" + filePath)
          .then(resp => resp.json())
          .then(jsonData => {
            const card = buildCardResumo(filePath, jsonData);
            dash.appendChild(card);
          })
          .catch(e => {
            console.error("Erro ao carregar:", filePath, e);
          });
      });
    })
    .catch(err => {
      console.error("Erro ao carregar main.json:", err);
      dash.textContent = "Não foi possível carregar main.json.";
    });
}

function listAllFiles(tree) {
  let results = [];
  if (tree.files) {
    tree.files.forEach(f => results.push(f));
  }
  for (const key in tree) {
    if (key !== "files") {
      results = results.concat(listAllFiles(tree[key]));
    }
  }
  return results;
}

function buildCardResumo(filePath, jsonData) {
  const card = document.createElement("div");
  card.classList.add("card");

  const assetName = extractAssetName(filePath);
  let probAl = 0, probNeu = 0, probQue = 0;
  let dataRelatorio = "Desconhecida";

  if (jsonData.relatorio_C?.probabilidade?.direcao) {
    const dirProb = jsonData.relatorio_C.probabilidade.direcao;
    probAl = parseFloat((dirProb.Alta || "0").replace("%", "")) || 0;
    probNeu = parseFloat((dirProb.Neutra || "0").replace("%", "")) || 0;
    probQue = parseFloat((dirProb.Queda || "0").replace("%", "")) || 0;
  }

  if (jsonData.relatorio_C?.data) {
    dataRelatorio = jsonData.relatorio_C.data;
  }

  // Define a cor do card com base na probabilidade dominante
  if (probAl >= probQue && probAl >= probNeu) {
    card.classList.add("alta");
  } else if (probQue >= probAl && probQue >= probNeu) {
    card.classList.add("queda");
  } else {
    card.classList.add("neutra");
  }

  const header = document.createElement("div");
  header.classList.add("card-header");
  header.textContent = assetName;
  card.appendChild(header);

  // Cria um gráfico dentro do card utilizando Chart.js
  const chartCanvas = document.createElement("canvas");
  chartCanvas.width = 280;
  chartCanvas.height = 100;
  card.appendChild(chartCanvas);

  setTimeout(() => createChart(chartCanvas, probAl, probNeu, probQue), 200);

  // Ao clicar no card, abre o modal com informações detalhadas
  card.addEventListener("click", () => {
    openModal(assetName, dataRelatorio, probAl, probNeu, probQue, jsonData);
  });

  return card;
}

function extractAssetName(filePath) {
  const parts = filePath.split("/");
  return parts.length >= 4 ? parts[3] : "AtivoDesconhecido";
}

function createChart(canvas, probAl, probNeu, probQue) {
  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ["Alta", "Neutra", "Queda"],
      datasets: [{
        data: [probAl, probNeu, probQue],
        backgroundColor: ["#00FF00", "#B6B6B6", "#FF0000"],
      }]
    },
    options: {
      responsive: false,
      legend: { display: true }
    }
  });
}

// Função para configurar o modal
function setupModal() {
  const modal = document.createElement("div");
  modal.id = "reportModal";
  modal.classList.add("modal");
  modal.innerHTML = `
    <div class="modal-content">
      <span class="close">&times;</span>
      <h2 id="modal-title">Detalhes do Relatório</h2>
      <table id="modal-table"></table>
    </div>
  `;
  document.body.appendChild(modal);

  // Evento para fechar o modal ao clicar no "X"
  document.querySelector(".close").addEventListener("click", () => {
    closeModal();
  });

  // Fecha o modal ao clicar fora da área de conteúdo
  window.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
}

// Função para abrir o modal com informações detalhadas
function openModal(assetName, dataRelatorio, probAl, probNeu, probQue, jsonData) {
  const modal = document.getElementById("reportModal");
  if (!modal) {
    console.error("Modal não encontrado!");
    return;
  }

  document.getElementById("modal-title").textContent = `Relatório de ${assetName}`;

  const table = document.getElementById("modal-table");
  if (!table) {
    console.error("Tabela não encontrada dentro do modal!");
    return;
  }

  // Conteúdo inicial do modal com as probabilidades do relatório
  let tableContent = `
    <tr><th>Data</th><td>${dataRelatorio}</td></tr>
    <tr><th>Alta</th><td>${probAl.toFixed(2)}%</td></tr>
    <tr><th>Neutra</th><td>${probNeu.toFixed(2)}%</td></tr>
    <tr><th>Queda</th><td>${probQue.toFixed(2)}%</td></tr>
  `;

  // Verifica se existem informações de gerenciamento de risco e as adiciona à tabela
  if (jsonData.gerenciamento_risco_global) {
    const gr = jsonData.gerenciamento_risco_global;
    const regime = gr.regime_global;
    const risk = gr.risk_management;

    tableContent += `
      <tr><th colspan="2" style="background-color: #333;">Gerenciamento de Risco</th></tr>
      <tr><th>Direção Global</th><td>${regime.direcao_global}</td></tr>
      <tr><th>Volatilidade Global</th><td>${regime.volatilidade_global}</td></tr>
      <tr><th>Interesse Global</th><td>${regime.interesse_global}</td></tr>
      <tr><th>Prob. Direção Global</th><td>${(regime.prob_direcao_global * 100).toFixed(2)}%</td></tr>
      <tr><th>Ação Recomendada</th><td>${risk.recommended_action}</td></tr>
      <tr><th>Tamanho Recomendado</th><td>${risk.recommended_size}</td></tr>
      <tr><th>Preço de Entrada</th><td>${risk.entry_price}</td></tr>
      <tr><th>VAR Esperado</th><td>${risk.expected_var}</td></tr>
    `;
  }

  table.innerHTML = tableContent;
  modal.style.display = "flex";
}

// Função para fechar o modal
function closeModal() {
  const modal = document.getElementById("reportModal");
  if (modal) {
    modal.style.display = "none";
  }
}
