document.addEventListener("DOMContentLoaded", () => {
  loadLatestReports();
});

/**
 * Carrega apenas o relatório mais recente para cada ativo,
 * e exibe no dashboard principal.
 */
function loadLatestReports() {
  const dash = document.getElementById("dashboard");
  dash.innerHTML = '<div class="loading">?? Carregando relatórios...</div>';

  fetch("frontend/json/main.json")
    .then(r => r.json())
    .then(data => {
      const allPaths = listAllFiles(data.reports);
      const grouped = groupReportsByAsset(allPaths);

      // Vamos criar para cada símbolo apenas um card com o relatório mais recente
      dash.innerHTML = "";
      const symbols = Object.keys(grouped);

      if (symbols.length === 0) {
        dash.textContent = "Nenhum relatório encontrado.";
        return;
      }

      // Para cada símbolo, buscar o último relatório
      symbols.forEach(symbol => {
        // Ordena ou filtra para achar o caminho do último arquivo
        // (aqui estamos assumindo que o último no array é o mais recente,
        //  mas é recomendável comparar datas se os nomes de arquivo contiverem data)
        const filePaths = grouped[symbol];
        const lastPath = filePaths[filePaths.length - 1];

        buildCardForSymbol(symbol, lastPath)
          .then(card => dash.appendChild(card))
          .catch(e => console.error("Erro ao criar card:", e));
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

function groupReportsByAsset(paths) {
  const map = {};
  paths.forEach(path => {
    const parts = path.split("/");
    const symbol = parts.length >= 2 ? parts[1] : "AtivoDesconhecido";
    if (!map[symbol]) map[symbol] = [];
    map[symbol].push(path);
  });
  // Opcional: Ordenar os paths de cada símbolo por data, se fizer sentido:
  // map[symbol].sort(...compararDatas...);
  return map;
}

/**
 * Constrói o card para o último relatório de um símbolo.
 * Carrega o JSON e exibe as informações relevantes.
 */
async function buildCardForSymbol(symbol, filePath) {
  const resp = await fetch("frontend/json/" + filePath);
  const jsonData = await resp.json();

  // Extrai dados
  let probAl = 0, probNeu = 0, probQue = 0;
  let dataRelatorio = "Desconhecida";
  if (jsonData.relatorio_C?.probabilidade?.direcao) {
    const d = jsonData.relatorio_C.probabilidade.direcao;
    probAl  = parseFloat((d.Alta || "0").replace("%", ""))   || 0;
    probNeu = parseFloat((d.Neutra || "0").replace("%", "")) || 0;
    probQue = parseFloat((d.Queda || "0").replace("%", ""))  || 0;
  }
  if (jsonData.relatorio_C?.data) {
    dataRelatorio = jsonData.relatorio_C.data;
  }

  const card = document.createElement("div");
  card.classList.add("card");

  // Definir cor do card
  if (probAl >= probQue && probAl >= probNeu) {
    card.classList.add("alta");
  } else if (probQue >= probAl && probQue >= probNeu) {
    card.classList.add("queda");
  } else {
    card.classList.add("neutra");
  }

  // Cabeçalho
  const header = document.createElement("div");
  header.classList.add("card-header");
  header.textContent = `${symbol} - Último Relatório`;
  card.appendChild(header);

  // Cria um gráfico
  const chartCanvas = document.createElement("canvas");
  chartCanvas.width = 280;
  chartCanvas.height = 100;
  card.appendChild(chartCanvas);

  setTimeout(() => createChart(chartCanvas, probAl, probNeu, probQue), 200);

  // Informações textuais
  const info = document.createElement("p");
  info.innerHTML = `
    <strong>Data:</strong> ${dataRelatorio}<br/>
    <strong>Alta:</strong> ${probAl.toFixed(2)}% - 
    <strong>Neutra:</strong> ${probNeu.toFixed(2)}% - 
    <strong>Queda:</strong> ${probQue.toFixed(2)}%
  `;
  card.appendChild(info);

  return card;
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
