// script.js

document.addEventListener("DOMContentLoaded", () => {
  loadAllReports();
  setupModal();
});

function parseReportDate(jsonData, filePath) {
  // Tenta extrair a data do conteúdo JSON
  if (jsonData.relatorio_C && jsonData.relatorio_C.data) {
    let dataStr = jsonData.relatorio_C.data;
    // Se o formato for "20250213T160054", converte para "2025-02-13T16:00:54"
    if (/^\d{8}T\d{6}$/.test(dataStr)) {
      dataStr = dataStr.replace(
        /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/,
        "$1-$2-$3T$4:$5:$6"
      );
    }
    return new Date(dataStr);
  }
  // Se não houver data no JSON, tenta extrair do nome do arquivo
  const regex = /(\d{8}T\d{6})/;
  const match = filePath.match(regex);
  if (match) {
    let dateStr = match[1].replace(
      /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/,
      "$1-$2-$3T$4:$5:$6"
    );
    return new Date(dateStr);
  }
  // Se não encontrar nenhuma data, retorna uma data padrão
  return new Date(0);
}

function loadAllReports() {
  const dash = document.getElementById("dashboard");
  dash.innerHTML = '<div class="loading">?? Carregando relatórios...</div>';

  fetch("frontend/json/main.json")
    .then(r => r.json())
    .then(data => {
      const paths = listAllFiles(data.reports);
      
      if (paths.length === 0) {
        dash.textContent = "Nenhum relatório encontrado.";
        return;
      }
      
      // Cria um array de promessas para carregar todos os relatórios
      const fetchPromises = paths.map(filePath =>
        fetch("frontend/json/" + filePath)
          .then(resp => resp.json())
          .then(jsonData => {
            // Extrai a data do relatório
            const reportDate = parseReportDate(jsonData, filePath);
            return { filePath, jsonData, reportDate };
          })
          .catch(e => {
            console.error("Erro ao carregar:", filePath, e);
            return null; // Filtra erros
          })
      );

      Promise.all(fetchPromises).then(results => {
        // Filtra os carregamentos que falharam
        const reports = results.filter(item => item !== null);
        // Ordena do mais recente para o mais antigo
        reports.sort((a, b) => b.reportDate - a.reportDate);

        // Agrupamento por ativo (símbolo)
        const groupedReports = {};
        reports.forEach(item => {
          const asset = extractAssetName(item.filePath);
          if (!groupedReports[asset]) {
            groupedReports[asset] = [];
          }
          groupedReports[asset].push(item);
        });

        // Limpa o dashboard e renderiza os grupos
        dash.innerHTML = "";
        for (const asset in groupedReports) {
          // Cria um cabeçalho para o grupo
          const groupHeader = document.createElement("h2");
          groupHeader.textContent = asset;
          dash.appendChild(groupHeader);

          // Cria um container para os cards deste ativo
          const groupContainer = document.createElement("div");
          groupContainer.classList.add("group-container");
          groupContainer.style.display = "flex";
          groupContainer.style.flexWrap = "wrap";
          groupContainer.style.gap = "20px";
          groupContainer.style.justifyContent = "center";

          // Renderiza cada relatório do grupo
          groupedReports[asset].forEach(item => {
            const card = buildCardResumo(item.filePath, item.jsonData);
            groupContainer.appendChild(card);
          });

          dash.appendChild(groupContainer);
        }
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

  // Adiciona a classe "recent" se o relatório for das últimas 24 horas
  const now = new Date();
  const reportDate = new Date(dataRelatorio);
  if (!isNaN(reportDate)) { // Verifica se a data é válida
    const diffHoras = (now - reportDate) / (1000 * 3600);
    if (diffHoras <= 24) {
      card.classList.add("recent");
    }
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
