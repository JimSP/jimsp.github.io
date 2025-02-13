// script.js

document.addEventListener("DOMContentLoaded", () => {
  loadAllReports();
  setupModal();
});

/**
 * Fun��o para extrair a data do relat�rio.
 * Tenta obter a data a partir do JSON ou, como fallback, extrair do nome do arquivo.
 */
function parseReportDate(jsonData, filePath) {
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
  // Fallback: extrair a data do nome do arquivo
  const regex = /(\d{8}T\d{6})/;
  const match = filePath.match(regex);
  if (match) {
    let dateStr = match[1].replace(
      /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/,
      "$1-$2-$3T$4:$5:$6"
    );
    return new Date(dateStr);
  }
  // Se n�o encontrar, retorna uma data padr�o (muito antiga)
  return new Date(0);
}

/**
 * Fun��o recursiva para listar todos os arquivos presentes na �rvore de diret�rios.
 */
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

/**
 * Extrai o nome do ativo a partir do caminho do arquivo.
 * Considera que a estrutura �: ano/mes/dia/s�mbolo/nome_arquivo.json
 */
function extractAssetName(filePath) {
  const parts = filePath.split("/");
  return parts.length >= 4 ? parts[3] : "AtivoDesconhecido";
}

/**
 * Cria um card resumido para cada relat�rio, configurando cores, gr�ficos e eventos.
 */
function buildCardResumo(filePath, jsonData) {
  const card = document.createElement("div");
  card.classList.add("card");

  const assetName = extractAssetName(filePath);
  let probAl = 0, probNeu = 0, probQue = 0;
  let dataRelatorio = "Desconhecida";

  if (jsonData.relatorio_C && jsonData.relatorio_C.probabilidade && jsonData.relatorio_C.probabilidade.direcao) {
    const dirProb = jsonData.relatorio_C.probabilidade.direcao;
    probAl = parseFloat((dirProb.Alta || "0").replace("%", "")) || 0;
    probNeu = parseFloat((dirProb.Neutra || "0").replace("%", "")) || 0;
    probQue = parseFloat((dirProb.Queda || "0").replace("%", "")) || 0;
  }

  if (jsonData.relatorio_C && jsonData.relatorio_C.data) {
    dataRelatorio = jsonData.relatorio_C.data;
  }

  // Adiciona a classe "recent" se o relat�rio for das �ltimas 24 horas
  const now = new Date();
  const reportDate = parseReportDate(jsonData, filePath);
  if (!isNaN(reportDate)) {
    const diffHours = (now - reportDate) / (1000 * 3600);
    if (diffHours <= 24) {
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

  // Cria o cabe�alho do card com o nome do ativo
  const header = document.createElement("div");
  header.classList.add("card-header");
  header.textContent = assetName;
  card.appendChild(header);

  // Cria o canvas para o gr�fico utilizando Chart.js
  const chartCanvas = document.createElement("canvas");
  chartCanvas.width = 280;
  chartCanvas.height = 100;
  card.appendChild(chartCanvas);

  setTimeout(() => createChart(chartCanvas, probAl, probNeu, probQue), 200);

  // Evento: ao clicar, abre o modal com detalhes do relat�rio
  card.addEventListener("click", () => {
    openModal(assetName, dataRelatorio, probAl, probNeu, probQue, jsonData);
  });

  return card;
}

/**
 * Fun��o que cria um gr�fico do tipo "doughnut" para exibir as probabilidades.
 */
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
      plugins: {
        legend: { display: true }
      }
    }
  });
}

/**
 * Configura o modal de exibi��o dos detalhes do relat�rio.
 */
function setupModal() {
  const modal = document.createElement("div");
  modal.id = "reportModal";
  modal.classList.add("modal");
  modal.innerHTML = `
    <div class="modal-content">
      <span class="close">&times;</span>
      <h2 id="modal-title">Detalhes do Relat�rio</h2>
      <table id="modal-table"></table>
    </div>
  `;
  document.body.appendChild(modal);

  // Fecha o modal ao clicar no "X"
  modal.querySelector(".close").addEventListener("click", () => {
    closeModal();
  });

  // Fecha o modal ao clicar fora da �rea de conte�do
  window.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
}

/**
 * Abre o modal exibindo os detalhes do relat�rio.
 */
function openModal(assetName, dataRelatorio, probAl, probNeu, probQue, jsonData) {
  const modal = document.getElementById("reportModal");
  if (!modal) {
    console.error("Modal n�o encontrado!");
    return;
  }

  document.getElementById("modal-title").textContent = `Relat�rio de ${assetName}`;
  const table = document.getElementById("modal-table");
  if (!table) {
    console.error("Tabela n�o encontrada dentro do modal!");
    return;
  }

  let tableContent = `
    <tr><th>Data</th><td>${dataRelatorio}</td></tr>
    <tr><th>Alta</th><td>${probAl.toFixed(2)}%</td></tr>
    <tr><th>Neutra</th><td>${probNeu.toFixed(2)}%</td></tr>
    <tr><th>Queda</th><td>${probQue.toFixed(2)}%</td></tr>
  `;

  // Verifica se h� informa��es de gerenciamento de risco e as adiciona
  if (jsonData.gerenciamento_risco_global) {
    const gr = jsonData.gerenciamento_risco_global;
    const regime = gr.regime_global;
    const risk = gr.risk_management;

    tableContent += `
      <tr><th colspan="2" style="background-color: #333;">Gerenciamento de Risco</th></tr>
      <tr><th>Dire��o Global</th><td>${regime.direcao_global}</td></tr>
      <tr><th>Volatilidade Global</th><td>${regime.volatilidade_global}</td></tr>
      <tr><th>Interesse Global</th><td>${regime.interesse_global}</td></tr>
      <tr><th>Prob. Dire��o Global</th><td>${(regime.prob_direcao_global * 100).toFixed(2)}%</td></tr>
      <tr><th>A��o Recomendada</th><td>${risk.recommended_action}</td></tr>
      <tr><th>Tamanho Recomendado</th><td>${risk.recommended_size}</td></tr>
      <tr><th>Pre�o de Entrada</th><td>${risk.entry_price}</td></tr>
      <tr><th>VAR Esperado</th><td>${risk.expected_var}</td></tr>
    `;
  }

  table.innerHTML = tableContent;
  modal.style.display = "flex";
}

/**
 * Fecha o modal.
 */
function closeModal() {
  const modal = document.getElementById("reportModal");
  if (modal) {
    modal.style.display = "none";
  }
}

/**
 * Fun��o principal para carregar, ordenar e agrupar os relat�rios.
 */
function loadAllReports() {
  const dash = document.getElementById("dashboard");
  dash.innerHTML = '<div class="loading">?? Carregando relat�rios...</div>';

  fetch("frontend/json/main.json")
    .then(response => response.json())
    .then(data => {
      const paths = listAllFiles(data.reports);
      if (paths.length === 0) {
        dash.textContent = "Nenhum relat�rio encontrado.";
        return;
      }

      // Cria um array de promessas para carregar cada relat�rio
      const fetchPromises = paths.map(filePath =>
        fetch("frontend/json/" + filePath)
          .then(resp => resp.json())
          .then(jsonData => {
            const reportDate = parseReportDate(jsonData, filePath);
            return { filePath, jsonData, reportDate };
          })
          .catch(e => {
            console.error("Erro ao carregar:", filePath, e);
            return null;
          })
      );

      Promise.all(fetchPromises).then(results => {
        const reports = results.filter(item => item !== null);
        // Ordena os relat�rios do mais recente para o mais antigo
        reports.sort((a, b) => b.reportDate - a.reportDate);

        // Agrupa os relat�rios por ativo (s�mbolo)
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
          const groupDiv = document.createElement("div");
          groupDiv.classList.add("group");

          const groupTitle = document.createElement("h2");
          groupTitle.classList.add("group-title");
          groupTitle.textContent = asset;
          groupDiv.appendChild(groupTitle);

          const groupContainer = document.createElement("div");
          groupContainer.classList.add("group-container");

          groupedReports[asset].forEach(item => {
            const card = buildCardResumo(item.filePath, item.jsonData);
            groupContainer.appendChild(card);
          });

          groupDiv.appendChild(groupContainer);
          dash.appendChild(groupDiv);
        }
      });
    })
    .catch(err => {
      console.error("Erro ao carregar main.json:", err);
      dash.textContent = "N�o foi poss�vel carregar main.json.";
    });
}
