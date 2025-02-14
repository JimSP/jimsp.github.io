document.addEventListener("DOMContentLoaded", () => {
  loadFullHistory();
});

function loadFullHistory() {
  const panel = document.getElementById("historyPanel");
  panel.innerHTML = '<div class="loading">?? Carregando histórico...</div>';

  fetch("json/main.json")
    .then(r => r.json())
    .then(data => {
      const allPaths = listAllFiles(data.reports);
      const grouped = groupReportsByAsset(allPaths);

      panel.innerHTML = "";
      const symbols = Object.keys(grouped);

      if (symbols.length === 0) {
        panel.textContent = "Nenhum relatório encontrado.";
        return;
      }

      symbols.forEach(symbol => {
        // Cria um bloco de histórico para cada símbolo
        buildHistoryForSymbol(symbol, grouped[symbol])
          .then(div => panel.appendChild(div))
          .catch(e => console.error("Erro ao gerar histórico:", e));
      });
    })
    .catch(err => {
      console.error("Erro ao carregar main.json:", err);
      panel.textContent = "Não foi possível carregar main.json.";
    });
}

// As mesmas funções de listAllFiles e groupReportsByAsset podem ser copiadas ou compartilhadas
function listAllFiles(tree) { /* ... */ }
function groupReportsByAsset(paths) { /* ... */ }

async function buildHistoryForSymbol(symbol, filePaths) {
  // Carrega todos os JSONs para este símbolo
  const reportsData = [];
  for (const path of filePaths) {
    try {
      const resp = await fetch("json/" + path);
      const jsonData = await resp.json();
      reportsData.push({ path, data: jsonData });
    } catch(e) {
      console.error("Erro ao carregar arquivo:", path, e);
    }
  }

  // Monta um container
  const container = document.createElement("div");
  container.classList.add("historico-simbolo");

  const title = document.createElement("h2");
  title.textContent = `Histórico de: ${symbol}`;
  container.appendChild(title);

  // Exemplo: criar uma tabela
  let tableHTML = `
    <table>
      <tr>
        <th>Arquivo</th>
        <th>Data</th>
        <th>Alta</th>
        <th>Neutra</th>
        <th>Queda</th>
      </tr>
  `;

  // Preencher tabela
  reportsData.forEach(({ path, data }) => {
    const relC = data.relatorio_C || {};
    const prob = relC.probabilidade?.direcao || {};
    const dtRel = relC.data || "Data Desconhecida";
    const pa = parseFloat((prob.Alta || "0").replace("%", "")) || 0;
    const pn = parseFloat((prob.Neutra || "0").replace("%", "")) || 0;
    const pq = parseFloat((prob.Queda || "0").replace("%", "")) || 0;

    tableHTML += `
      <tr>
        <td>${path}</td>
        <td>${dtRel}</td>
        <td>${pa.toFixed(2)}%</td>
        <td>${pn.toFixed(2)}%</td>
        <td>${pq.toFixed(2)}%</td>
      </tr>
    `;
  });

  tableHTML += `</table>`;
  container.innerHTML += tableHTML;
  return container;
}
