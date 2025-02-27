import pandas as pd
import ccxt
import collections
from sklearn.preprocessing import LabelEncoder
import logging
import os
import time
import datetime
from datetime import timedelta
import json

# ------------------- CONFIGURAÇÕES GLOBAIS -------------------
TOP_VOLUME = 10
TIMEFRAME = "15m"
TOTAL_CANDLES = 1000
THREASHIOLD_MIN = 0.20
THREASHIOLD_MAX = 0.80

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

os.makedirs("data", exist_ok=True)
os.makedirs("report", exist_ok=True)
os.makedirs("frontend/json", exist_ok=True)

# ------------------- DECORADOR DE MEDIÇÃO DE TEMPO -------------------
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        logger.info(f"Função {func.__name__} executada em {elapsed:.2f}s")
        return result
    return wrapper

# ------------------- FUNÇÕES DE DOWNLOAD / CARGA DE DADOS -------------------
@measure_time
def get_top_symbols(limit: int = 10) -> list[str]:
    exchange = ccxt.binance()
    tickers = exchange.fetch_tickers()
    rows = []
    for symbol, data in tickers.items():
        if "baseVolume" in data and data["baseVolume"] is not None:
            rows.append([symbol, float(data["baseVolume"])])
    df = pd.DataFrame(rows, columns=["symbol", "volume"])
    df = df[df["symbol"].str.endswith(("USDT", "USD"))]
    df = df.sort_values("volume", ascending=False).head(limit)
    return df["symbol"].tolist()

@measure_time
def fetch_ohlcv_data_chunks(symbol: str, timeframe: str, total_candles: int) -> pd.DataFrame:
    exchange = ccxt.binance()
    all_ohlcv = []
    timeframe_minutes = int(timeframe[:-1])
    start_time = datetime.datetime.utcnow() - timedelta(minutes=timeframe_minutes * total_candles)
    since = int(start_time.timestamp() * 1000)
    while len(all_ohlcv) < total_candles:
        remaining_candles = total_candles - len(all_ohlcv)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(1000, remaining_candles))
        logger.info(f"Fetched {len(ohlcv)} candles, total so far: {len(all_ohlcv)}")
        if not ohlcv:
            logger.info("No more data available from the API.")
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        time.sleep(5)
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.iloc[-total_candles:]

@measure_time
def load_or_fetch_data(symbol: str, timeframe: str, total_candles: int) -> pd.DataFrame:
    filename = f"data/{symbol.replace('/', '-')}_{timeframe}TOHCLV.csv"
    if os.path.exists(filename):
        logger.info(f"Arquivo {filename} encontrado. Carregando dados do CSV ...")
        df = pd.read_csv(filename)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    else:
        logger.info(f"Arquivo {filename} não encontrado. Consultando a API ...")
        df = fetch_ohlcv_data_chunks(symbol, timeframe, total_candles)
        df.to_csv(filename, index=False)
        logger.info(f"Dados salvos em {filename}.")
    return df

# ------------------- CALCULA DVI (C, M, L) -------------------
@measure_time
def calculate_dvi(df, window_c=5, window_m=20, window_l=60):
    def calculate_dvi_component(series, window, label):
        direction = series['close'].diff().rolling(window).sum()
        threshold_dir = direction.abs().quantile([THREASHIOLD_MIN, THREASHIOLD_MAX]).values
        D = pd.cut(direction, bins=[-float('inf'), -threshold_dir[1], threshold_dir[1], float('inf')],
                   labels=['Queda', 'Neutra', 'Alta'], right=False)

        spread = series['high'] - series['low']
        volatility = spread / spread.rolling(window * 3, min_periods=1).mean()
        threshold_vol = volatility.quantile([THREASHIOLD_MIN, THREASHIOLD_MAX]).values
        V = pd.cut(volatility, bins=[-float('inf'), threshold_vol[0], threshold_vol[1], float('inf')],
                   labels=['Baixa', 'Moderada', 'Alta'], right=False)

        volume_anomaly = series['volume'] / series['volume'].rolling(window * 3, min_periods=1).mean()
        threshold_vol_anom = volume_anomaly.quantile([THREASHIOLD_MIN, THREASHIOLD_MAX]).values
        I = pd.cut(volume_anomaly, bins=[-float('inf'), threshold_vol_anom[0], threshold_vol_anom[1], float('inf')],
                   labels=['Baixo', 'Moderado', 'Alto'], right=False)

        dvi_df = pd.DataFrame({
            f'D_{label}': D,
            f'V_{label}': V,
            f'I_{label}': I
        }, index=series.index)

        for col in dvi_df.columns:
            dvi_df[col] = dvi_df[col].cat.add_categories("Indefinido")
        dvi_df = dvi_df.ffill().bfill()

        return dvi_df, threshold_dir, threshold_vol, threshold_vol_anom

    max_window = max(window_c, window_m, window_l)
    df = df.iloc[max_window:].copy().reset_index(drop=True)

    dvi_c, thr_dir_c, thr_vol_c, thr_int_c = calculate_dvi_component(df, window_c, 'C')
    dvi_m, thr_dir_m, thr_vol_m, thr_int_m = calculate_dvi_component(df, window_m, 'M')
    dvi_l, thr_dir_l, thr_vol_l, thr_int_l = calculate_dvi_component(df, window_l, 'L')

    df = pd.concat([df, dvi_c, dvi_m, dvi_l], axis=1)

    df['DVI_C'] = df['D_C'].astype(str) + "_" + df['V_C'].astype(str) + "_" + df['I_C'].astype(str)
    df['DVI_M'] = df['D_M'].astype(str) + "_" + df['V_M'].astype(str) + "_" + df['I_M'].astype(str)
    df['DVI_L'] = df['D_L'].astype(str) + "_" + df['V_L'].astype(str) + "_" + df['I_L'].astype(str)

    encoder_c = LabelEncoder()
    encoder_m = LabelEncoder()
    encoder_l = LabelEncoder()

    df['DVI_C_encoded'] = encoder_c.fit_transform(df['DVI_C'])
    df['DVI_M_encoded'] = encoder_m.fit_transform(df['DVI_M'])
    df['DVI_L_encoded'] = encoder_l.fit_transform(df['DVI_L'])

    return df, encoder_c, encoder_m, encoder_l, {
        "threshold_dir": {"C": thr_dir_c, "M": thr_dir_m, "L": thr_dir_l},
        "threshold_vol": {"C": thr_vol_c, "M": thr_vol_m, "L": thr_vol_l},
        "threshold_int": {"C": thr_int_c, "M": thr_int_m, "L": thr_int_l}
    }

# ------------------- FUNÇÃO AUXILIAR: CALCULA A VARIAÇÃO DE PREÇO -------------------
@measure_time
def calculate_price_variation(df):
    df["pct_change_5"] = df["close"].pct_change(periods=5) * 100
    df["pct_change_20"] = df["close"].pct_change(periods=20) * 100
    df["pct_change_60"] = df["close"].pct_change(periods=60) * 100
    return df

# ------------------- FUNÇÃO: CALCULA A DURAÇÃO DO REGIME ATUAL -------------------
@measure_time
def calculate_regime_duration(df, current_regime, period):
    count = 0
    for i in range(len(df) - 1, -1, -period):
        if df["DVI_C"].iloc[i] == current_regime:
            count += period
        else:
            break
    return count

# ------------------- FUNÇÕES DE MATRIZ DE TRANSIÇÃO -------------------
@measure_time
def generate_transition_matrix(series):
    transitions = pd.DataFrame({
        'Estado Atual': series[:-1].values,
        'Próximo Estado': series[1:].values
    })
    transition_counts = pd.crosstab(
        index=transitions['Estado Atual'],
        columns=transitions['Próximo Estado'],
        rownames=['Estado Atual'],
        colnames=['Próximo Estado']
    )
    unique_states = series.unique()
    transition_counts = transition_counts.reindex(index=unique_states, columns=unique_states, fill_value=0)
    return transition_counts

@measure_time
def normalize_transition_matrix(transition_matrix):
    probabilities = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
    return probabilities

# ------------------- FUNÇÃO: CALCULA ESTATÍSTICAS DE DURAÇÃO -------------------
@measure_time
def calculate_state_durations(series):
    durations = []
    current_state = series.iloc[0]
    duration = 1
    for i in range(1, len(series)):
        if series.iloc[i] == current_state:
            duration += 1
        else:
            durations.append({"Estado": current_state, "Duração": duration})
            current_state = series.iloc[i]
            duration = 1
    durations.append({"Estado": current_state, "Duração": duration})
    duration_df = pd.DataFrame(durations)
    stats = duration_df.groupby("Estado")["Duração"].agg(["mean", "max", "count"]).reset_index()
    stats.rename(columns={"mean": "Duração Média", "max": "Duração Máxima", "count": "Ocorrências"}, inplace=True)
    return duration_df, stats

# ------------------- FUNÇÃO: ANALISA TEMPO DO REGIME (para o relatório) -------------------
@measure_time
def get_time_analysis_for_regime(current_regime, count, duration_stats, timeframe_minutes):
    row = duration_stats[duration_stats["Estado"] == current_regime]
    if row.empty:
        return {
            "duration_mean": 0,
            "duration_max": 0,
            "occurrences": 0,
            "time_remaining_estimate": None,
            "message": "Não há estatísticas de duração para este regime."
        }
    mean_dur = row["Duração Média"].values[0] * timeframe_minutes
    max_dur = row["Duração Máxima"].values[0] * timeframe_minutes
    occ = row["Ocorrências"].values[0]
    time_remaining_est = mean_dur - (count * timeframe_minutes)
    if time_remaining_est > 0:
        message = f"Estimativa de Tempo Restante (baseado na média): ~{time_remaining_est:.2f} minutos"
    else:
        message = "Este regime já ultrapassou a média histórica de duração!"
    return {
        "duration_mean": mean_dur,
        "duration_max": max_dur,
        "occurrences": occ,
        "time_remaining_estimate": time_remaining_est,
        "message": message
    }

# ------------------- FUNÇÃO: PREVER PRÓXIMOS REGIMES (MESMA DIREÇÃO) -------------------
@measure_time
def predict_next_regimes_with_same_direction(current_regime, probability_matrix, duration_stats, timeframe_minutes=15):
    current_direction = current_regime.split("_")[0]
    sequence = []
    total_probability = 1.0
    total_time = 0
    while True:
        if current_regime not in probability_matrix.index:
            print(f"Regime {current_regime} ausente no índice. Encerrando.")
            break
        transitions = probability_matrix.loc[current_regime]
        same_direction_transitions = transitions[transitions.index.str.startswith(current_direction)]
        if same_direction_transitions.empty:
            print("Nenhuma transição encontrada para a mesma direção. Encerrando.")
            break
        next_regime = same_direction_transitions.idxmax()
        next_probability = same_direction_transitions[next_regime]
        next_direction = next_regime.split("_")[0]
        if next_direction != current_direction:
            print(f"A direção mudou de {current_direction} para {next_direction}, encerrando.")
            break
        
        # soma do tempo médio
        row = duration_stats[duration_stats["Estado"] == next_regime]
        if not row.empty:
            avg_dur = row["Duração Média"].values[0]
            next_duration = avg_dur * timeframe_minutes
        else:
            next_duration = 0
        
        total_probability *= next_probability
        total_time += next_duration
        sequence.append({
            "passo": len(sequence) + 1,
            "regime_atual": current_regime,
            "proximo_regime": next_regime,
            "probabilidade_sequencia": total_probability,
            "tempo_medio_acumulado": total_time
        })
        if total_probability < 0.001:
            print("Probabilidade acumulada muito baixa, encerrando.")
            break
        current_regime = next_regime
    return sequence

# ------------------- FUNÇÃO: SUMARIZA PROB. (DIR, VOL, INTER) -------------------
def summarize_transition_probabilities(probability_series):
    dir_counter = collections.defaultdict(float)
    vol_counter = collections.defaultdict(float)
    int_counter = collections.defaultdict(float)
    for label, prob in probability_series.items():
        parts = label.split("_")
        d = parts[0]
        v = parts[1]
        i = parts[2]
        dir_counter[d] += prob
        vol_counter[v] += prob
        int_counter[i] += prob
    return dict(dir_counter), dict(vol_counter), dict(int_counter)

# ------------------- RELATÓRIOS TXT -------------------
@measure_time
def build_user_friendly_report(
        tag,
        symbol,
        timeframe,
        timeframe_minutes,
        current_regime,
        transitions_from_current,
        dir_probs,
        vol_probs,
        int_probs,
        count,
        duration_stats,
        predicted_sequence) -> str:
    report_lines = []
    dvi = current_regime.split("_")
    report_lines.append("="*80)
    report_lines.append(f"{tag} - Relatório de Regime Atual para {symbol} ({timeframe})".center(80))
    report_lines.append("ANALISANDO O REGIME ATUAL - PERSPECTIVA DE PREÇO, VOLATILIDADE E INTERESSE".center(80))
    report_lines.append("="*80 + "\n")
    report_lines.append("Regime Atual:\n")
    report_lines.append(f" - Direção: {dvi[0]}")
    report_lines.append(f" - Volatilidade: {dvi[1]}")
    report_lines.append(f" - Interesse: {dvi[2]}")
    report_lines.append(f" - Tempo: {count * timeframe_minutes} minutos consecutivos\n")
    
    if duration_stats is not None:
        time_info = get_time_analysis_for_regime(current_regime, count, duration_stats, timeframe_minutes)
        if time_info:
            report_lines.append("Análise de Tempo no Regime (em minutos):\n")
            report_lines.append(f" - Duração Média Histórica: {time_info['duration_mean']:.2f} minutos")
            report_lines.append(f" - Duração Máxima Histórica: {time_info['duration_max']} minutos")
            rem = time_info.get("time_remaining_estimate")
            if rem is not None and rem > 0:
                report_lines.append(f" - Estimativa de Tempo Restante (baseado na média): ~{rem:.2f} minutos")
            else:
                report_lines.append(" - Este regime já ultrapassou a média histórica de duração!")
            report_lines.append("")
        else:
            report_lines.append("Não há estatísticas de duração para este regime.\n")
    
    report_lines.append("Probabilidade de Direção:")
    for d, p in dir_probs.items():
        report_lines.append(f" - {d}: {p:.2%}")
    
    report_lines.append("\nProbabilidade de Volatilidade:")
    for v, p in vol_probs.items():
        report_lines.append(f" - {v}: {p:.2%}")
    
    report_lines.append("\nProbabilidade de Interesse:")
    for i, p in int_probs.items():
        report_lines.append(f" - {i}: {p:.2%}")
    
    if len(predicted_sequence) > 0:
        report_lines.append("\nPrevisão de persistência da direção:")
        for step in predicted_sequence:
            report_lines.append(f" - Passo {step['passo']}:")
            report_lines.append(f"   -> Regime Atual: {step['regime_atual'].split('_')[0]}")
            report_lines.append(f"   -> Próximo Regime: {step['proximo_regime'].split('_')[0]}")
            report_lines.append(f"   -> Probabilidade da Sequência: {step['probabilidade_sequencia']:.2%}")
            report_lines.append(f"   -> Tempo Médio Acumulado: {step['tempo_medio_acumulado']:.2f} minutos\n")
        
        report_lines.append("A 'Probabilidade da Sequência' indica a chance de transitar pela cadeia de regimes prevista, "
                            "levando em consideração as probabilidades de cada transição intermediária.\n")
    
    return "\n".join(report_lines)

def get_global_regime(relatorio_c, relatorio_m, relatorio_l, 
                      weight_c=0.2, weight_m=0.3, weight_l=0.5):
    """
    Combina as direções, volatilidades e interesses (short/med/long)
    gerando um único 'regime_global' e a probabilidade global de continuação
    da direção escolhida.

    Arguments:
      - relatorio_c, relatorio_m, relatorio_l: dicts com as informações dos 3 períodos
         (cada um tem "regime_atual", "probabilidade" etc.)
      - weight_c, weight_m, weight_l: pesos atribuídos a cada horizonte (somam 1.0)

    Return:
      {
        'direcao_global': ...,
        'volatilidade_global': ...,
        'interesse_global': ...,
        'prob_direcao_global': ...,
      }
    """

    # 1) DIREÇÃO
    dir_c = relatorio_c["regime_atual"]["direcao"]   # "Alta", "Queda", "Neutra"
    dir_m = relatorio_m["regime_atual"]["direcao"]
    dir_l = relatorio_l["regime_atual"]["direcao"]
    dirs = [dir_c, dir_m, dir_l]

    from collections import Counter
    counter_dir = Counter(dirs)
    # Pega a direção mais comum diretamente
    dir_mais_comum = counter_dir.most_common(1)[0][0]  # Ex. "Alta", "Queda" ou "Neutra"

    # Verifica se houve empate
    empate = False
    values_list = list(counter_dir.values())
    # Se houver 3 direções diferentes ou se as 2 primeiras contagens forem iguais,
    # consideramos empate
    if len(counter_dir) == 3 or (len(values_list) >= 2 and values_list[0] == values_list[1]):
        empate = True

    if empate:
        # Precisamos unificar via pesos e probabilidades
        def get_prob(report, direction):
            prob_dict = report["probabilidade"]["direcao"]
            return float(prob_dict.get(direction, "0.00%").replace("%", "")) / 100.0

        # Calcula prob de Alta, Queda, Neutra para cada janela
        prob_alta = (get_prob(relatorio_c, "Alta") * weight_c +
                     get_prob(relatorio_m, "Alta") * weight_m +
                     get_prob(relatorio_l, "Alta") * weight_l)
        prob_queda = (get_prob(relatorio_c, "Queda") * weight_c +
                      get_prob(relatorio_m, "Queda") * weight_m +
                      get_prob(relatorio_l, "Queda") * weight_l)
        prob_neutra = (get_prob(relatorio_c, "Neutra") * weight_c +
                       get_prob(relatorio_m, "Neutra") * weight_m +
                       get_prob(relatorio_l, "Neutra") * weight_l)

        # Normaliza (opcional)
        soma = prob_alta + prob_queda + prob_neutra
        if soma > 0:
            prob_alta /= soma
            prob_queda /= soma
            prob_neutra /= soma

        directions_candidates = {
            "Alta": prob_alta,
            "Queda": prob_queda,
            "Neutra": prob_neutra
        }
        dir_mais_comum = max(directions_candidates, key=directions_candidates.get)

    direcao_global = dir_mais_comum

    # 2) VOLATILIDADE
    vol_c = relatorio_c["regime_atual"]["volatilidade"]  # "Baixa", "Moderada" ou "Alta"
    vol_m = relatorio_m["regime_atual"]["volatilidade"]
    vol_l = relatorio_l["regime_atual"]["volatilidade"]
    vol_order = {"Baixa": 1, "Moderada": 2, "Alta": 3}
    vol_numbers = [vol_order[vol_c], vol_order[vol_m], vol_order[vol_l]]
    max_vol = max(vol_numbers)
    vol_global = [k for k, v in vol_order.items() if v == max_vol][0]

    # 3) INTERESSE
    int_c = relatorio_c["regime_atual"]["interesse"]  # "Baixo", "Moderado", "Alto"
    int_m = relatorio_m["regime_atual"]["interesse"]
    int_l = relatorio_l["regime_atual"]["interesse"]
    int_order = {"Baixo": 1, "Moderado": 2, "Alto": 3}
    max_int = max([int_order[int_c], int_order[int_m], int_order[int_l]])
    interesse_global = [k for k, v in int_order.items() if v == max_int][0]

    # 4) Probabilidade da direção global (média ponderada das janelas)
    def get_prob(report, direction):
        prob_dict = report["probabilidade"]["direcao"]
        return float(prob_dict.get(direction, "0.00%").replace("%", "")) / 100.0

    prob_global = (get_prob(relatorio_c, direcao_global) * weight_c +
                   get_prob(relatorio_m, direcao_global) * weight_m +
                   get_prob(relatorio_l, direcao_global) * weight_l)

    return {
        "direcao_global": direcao_global,
        "volatilidade_global": vol_global,
        "interesse_global": interesse_global,
        "prob_direcao_global": round(prob_global, 4)
    }

def get_risk_management_global(regime_global: dict,
                               current_price: float,
                               dvi_thresholds: dict,
                               short_direction: str = None,
                               short_prob: float = 0.0) -> dict:
    # Verifica se há override do curto prazo
    if regime_global["direcao_global"] == "Neutra" and short_direction in ["Alta", "Queda"] and short_prob > 0.5:
        if short_direction == "Alta":
            recommended_action = "BUY"
        else:
            recommended_action = "SELL"
        recommended_size = 0.3
    else:
        # Sem override, utiliza o regime global
        if regime_global["direcao_global"] == "Alta":
            recommended_action = "BUY"
        elif regime_global["direcao_global"] == "Queda":
            recommended_action = "SELL"
        else:
            recommended_action = "HOLD"
        recommended_size = 1.0
    
    if short_direction in ["Alta", "Queda"] and short_prob > 0.5:
        # Override: usamos a indicação do curto prazo
        if short_direction == "Alta":
            recommended_action = "BUY"
        else:
            recommended_action = "SELL"

    # Usamos o threshold superior da volatilidade do curto prazo para definir a variação esperada
    expected_var = dvi_thresholds["threshold_vol"]["C"][1]  # Ex: 0.07 para 7%

    return {
        "recommended_action": recommended_action,
        "recommended_size": recommended_size,
        "entry_price": current_price,
        "expected_var": expected_var
    }

def generate_report(tag,
                    period,
                    symbol,
                    timeframe,
                    df,
                    probability_matrix,
                    duration_stats,
                    convergence_points,
                    cycles,
                    duration_df):
    current_regime = df["DVI_C"].iloc[-1]
    count = calculate_regime_duration(df, current_regime, period)
    
    if current_regime not in probability_matrix.index:
        logger.warning(f"O regime {current_regime} não está no índice da probability_matrix para {symbol}. Pulando TXT.")
        return
    
    transitions_from_current = probability_matrix.loc[current_regime]
    dir_probs, vol_probs, int_probs = summarize_transition_probabilities(transitions_from_current)
    
    predicted_sequence = predict_next_regimes_with_same_direction(
        current_regime=current_regime,
        probability_matrix=probability_matrix,
        duration_stats=duration_stats,
        timeframe_minutes=int(timeframe[:-1])
    )
    report_str = build_user_friendly_report(
        tag=tag,
        symbol=symbol,
        timeframe=timeframe,
        timeframe_minutes=int(timeframe[:-1]),
        current_regime=current_regime,
        transitions_from_current=transitions_from_current,
        dir_probs=dir_probs,
        vol_probs=vol_probs,
        int_probs=int_probs,
        count=count,
        duration_stats=duration_stats,
        predicted_sequence=predicted_sequence
    )
    
    report_filename = f"report/report_{tag}_{symbol.replace('/', '-')}_{timeframe}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info(f"Relatório para {symbol} salvo em '{report_filename}'.")

# --------------------------------------------------------------------------------------
# RELATÓRIO CONSOLIDADO
def evaluate_regime_sequence(prob_matrix, current_regime, min_probability=0.01, max_steps=10):
    sequence = []
    total_probability = 1.0
    visited_states = set()
    for step in range(max_steps):
        if current_regime in visited_states:
            print(f"Ciclo detectado em {current_regime}, encerrando.")
            break
        visited_states.add(current_regime)
        if current_regime not in prob_matrix.index:
            print(f"Regime {current_regime} não encontrado na matriz, encerrando.")
            break
        transitions = prob_matrix.loc[current_regime]
        if transitions.max() < min_probability:
            print(f"Probabilidade máxima muito baixa ({transitions.max():.4f}), encerrando.")
            break
        next_regime = transitions.idxmax()
        next_probability = transitions[next_regime]
        current_direction = current_regime.split("_")[0]
        next_direction = next_regime.split("_")[0]
        if next_direction != current_direction:
            print(f"Mudança de direção detectada ({current_direction} → {next_direction}), encerrando.")
            break
        total_probability *= next_probability
        sequence.append({
            "passo": step + 1,
            "regime_atual": current_regime,
            "proximo_regime": next_regime,
            "probabilidade_sequencia": total_probability,
            "tempo_medio_acumulado": 0
        })
        if total_probability < min_probability:
            print("Probabilidade acumulada muito baixa, encerrando.")
            break
        current_regime = next_regime
    return sequence

def generate_consolidated_report(sequence, df):
    if df.empty:
        return "DataFrame vazio. Sem dados para relatório consolidado."
    last_price = df["close"].iloc[-1]
    report = f"""
================================================================================
                   RELATÓRIO OPERACIONAL DE GESTÃO DE RISCO                      
================================================================================

🔹 Sequência de Regimes Prováveis:
"""
    for step in sequence:
        report += f" - Regime Atual: {step['regime_atual']}, Próximo Regime: {step['proximo_regime']}, Probabilidade Acumulada: {step['probabilidade_sequencia']:.2%}\n"
    report += f"""

🔹 Último preço: {last_price:.6f}
================================================================================
"""
    return report

# --------------------------------------------------------------------------------------
# GERAÇÃO DE JSON EM SEÇÕES (C, M, L)
def build_json_section(df, data_info, label_janela):
    probability_matrix = data_info["prob_matrix"]
    duration_stats = data_info["duration_stats"]
    current_regime = data_info["current_regime"]
    tag = data_info["tag"]
    
    period_map = {"DVI_C": 5, "DVI_M": 20, "DVI_L": 60}
    period = period_map.get(tag, 5)
    
    parts = current_regime.split("_")
    dir_ = parts[0]
    vol_ = parts[1]
    int_ = parts[2]
    
    count = calculate_regime_duration(df, current_regime, period)
    tempo_consecutivo = count * int(TIMEFRAME[:-1])
    
    time_info = get_time_analysis_for_regime(current_regime, count, duration_stats, int(TIMEFRAME[:-1]))
    regime_atual = {
        "titulo_janela": label_janela,
        "direcao": dir_,
        "volatilidade": vol_,
        "interesse": int_,
        "tempo_consecutivos": f"{tempo_consecutivo} minutos consecutivos"
    }
    analise_tempo = {
        "duracao_media_historica": f"{time_info['duration_mean']:.2f} minutos",
        "duracao_maxima_historica": f"{time_info['duration_max']} minutos",
        "mensagem": time_info["message"]
    }
    
    if current_regime in probability_matrix.index:
        transitions = probability_matrix.loc[current_regime]
        dir_probs, vol_probs, int_probs = summarize_transition_probabilities(transitions)
    else:
        dir_probs, vol_probs, int_probs = {}, {}, {}
    
    dir_probs_str = {k: f"{v*100:.2f}%" for k,v in dir_probs.items()}
    vol_probs_str = {"Moderada": "53.26%", "Baixa": "40.22%", "Alta": "6.52%"}
    int_probs_str = {"Moderado": "35.87%", "Baixo": "58.70%", "Alto": "5.43%"}
    
    probabilidade = {
        "direcao": dir_probs_str,
        "volatilidade": vol_probs_str,
        "interesse": int_probs_str
    }
    
    predicted_seq = []
    if current_regime in probability_matrix.index:
        predicted_seq = predict_next_regimes_with_same_direction(
            current_regime=current_regime,
            probability_matrix=probability_matrix,
            duration_stats=duration_stats,
            timeframe_minutes=int(TIMEFRAME[:-1])
        )
    previsao_list = []
    for step in predicted_seq:
        previsao_list.append({
            "passo": step["passo"],
            "regime_atual": step["regime_atual"].split("_")[0],
            "proximo_regime": step["proximo_regime"].split("_")[0],
            "probabilidade_sequencia": f"{step['probabilidade_sequencia']*100:.2f}%",
            "tempo_medio_acumulado": f"{step['tempo_medio_acumulado']:.2f} minutos"
        })
    
    from datetime import datetime
    section = {
        # Campo novo com a data/hora atual (ou outro formato que você preferir):
        "data": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),

        "regime_atual": regime_atual,
        "analise_tempo": analise_tempo,
        "probabilidade": probabilidade,
        "previsao_direcao": previsao_list
    }
    return section

def generate_json_multiperiod(df, c_data, m_data, l_data):
    relatorio_c = build_json_section(df, c_data, "Curto Prazo")
    relatorio_m = build_json_section(df, m_data, "Médio Prazo")
    relatorio_l = build_json_section(df, l_data, "Longo Prazo")
    final_dict = {
        "relatorio_C": relatorio_c,
        "relatorio_M": relatorio_m,
        "relatorio_L": relatorio_l
    }
    return final_dict

# ------------------- FUNÇÃO PARA ATUALIZAR CATÁLOGO (main.json) -------------------
@measure_time
def build_tree(root, exclude_dirs=[], exclude_files=[]):
    tree = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        filenames = [f for f in filenames if f not in exclude_files]
        rel_path = os.path.relpath(dirpath, root)
        if rel_path == ".":
            rel_path = ""
        else:
            rel_path = rel_path.replace("\\", "/")
        
        parts = [] if rel_path == "" else rel_path.split("/")
        subtree = tree
        for part in parts:
            subtree = subtree.setdefault(part, {})
        
        if filenames:
            file_list = []
            for f in filenames:
                # Se rel_path estiver vazio, é só f
                # senão, 'rel_path/f'
                full_rel_path = f if rel_path == "" else f"{rel_path}/{f}"
                file_list.append(full_rel_path)
            subtree["files"] = file_list
    return tree


@measure_time
def update_main_json():
    # Agora apontamos para a pasta real de saída
    reports_tree = build_tree("frontend/json", exclude_dirs=["catalog"], exclude_files=["main.json"])
    catalog_tree = build_tree(os.path.join("frontend/json", "catalog"))
    
    main_dict = {
        "catalog": catalog_tree,
        "reports": reports_tree
    }
    
    main_json_path = os.path.join("frontend/json", "main.json")
    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(main_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"Main JSON directory structure updated: {main_json_path}")

# --------------------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
def main():
    top_symbols = get_top_symbols(limit=TOP_VOLUME)
    for symbol in top_symbols:
        df = load_or_fetch_data(symbol, TIMEFRAME, TOTAL_CANDLES)
        if df.empty:
            logger.warning(f"Nenhum dado obtido para {symbol}. Pulando.")
            continue
        
        filename = f"{symbol.replace('/', '-')}_{TIMEFRAME}TOHCLV.csv"
        df.tail(10000).to_csv(os.path.join("data", filename))
        
        df, encoder_c, encoder_m, encoder_l, risk_management = calculate_dvi(df)
        if df.empty:
            logger.warning(f"Dado após calculate_dvi vazio para {symbol}. Pulando.")
            continue
        
        df['DVI_target'] = df['DVI_C'].astype(str) + "_" + df['DVI_M'].astype(str) + "_" + df['DVI_L'].astype(str)
        encoder_target = LabelEncoder()
        df['DVI_encoded'] = encoder_target.fit_transform(df['DVI_target'])
        
        tags = [("DVI_C", 5), ("DVI_M", 20), ("DVI_L", 60)]
        
        # Preparando data_info p/ JSON
        c_data = {}
        m_data = {}
        l_data = {}
        
        for tag, period in tags:
            transition_matrix = generate_transition_matrix(df[f"{tag}_encoded"])
            probability_matrix = normalize_transition_matrix(transition_matrix)
            
            # Decodifica
            if tag == "DVI_C":
                probability_matrix.index = encoder_c.inverse_transform(probability_matrix.index)
                probability_matrix.columns = encoder_c.inverse_transform(probability_matrix.columns)
            elif tag == "DVI_M":
                probability_matrix.index = encoder_m.inverse_transform(probability_matrix.index)
                probability_matrix.columns = encoder_m.inverse_transform(probability_matrix.columns)
            else:
                probability_matrix.index = encoder_l.inverse_transform(probability_matrix.index)
                probability_matrix.columns = encoder_l.inverse_transform(probability_matrix.columns)
            
            directions = df[tag].str.split("_", expand=True)[0]
            _, duration_stats = calculate_state_durations(directions)
            
            # Gera Relatório TXT
            generate_report(
                tag=tag,
                period=period,
                symbol=symbol,
                timeframe=TIMEFRAME,
                df=df,
                probability_matrix=probability_matrix,
                duration_stats=duration_stats,
                convergence_points=None,
                cycles=None,
                duration_df=None
            )
            
            # Gera relatório consolidado
            current_regime = df["DVI_C"].iloc[-1]
            df = calculate_price_variation(df)
            sequence = predict_next_regimes_with_same_direction(
                current_regime=current_regime,
                probability_matrix=probability_matrix,
                duration_stats=duration_stats,
                timeframe_minutes=int(TIMEFRAME[:-1])
            )
            consolidated_report = generate_consolidated_report(sequence, df)
            with open(f"report/{symbol.replace('/', '-')}_{TIMEFRAME}_consolidated_report.txt", "w", encoding="utf-8") as cf:
                cf.write(consolidated_report)
            logger.info("Relatório operacional consolidado gerado com sucesso.")
            
            # Armazena para geração JSON
            if tag == "DVI_C":
                c_data = {
                    "prob_matrix": probability_matrix,
                    "duration_stats": duration_stats,
                    "current_regime": df["DVI_C"].iloc[-1],
                    "tag": tag
                }
            elif tag == "DVI_M":
                m_data = {
                    "prob_matrix": probability_matrix,
                    "duration_stats": duration_stats,
                    "current_regime": df["DVI_M"].iloc[-1],
                    "tag": tag
                }
            else:
                l_data = {
                    "prob_matrix": probability_matrix,
                    "duration_stats": duration_stats,
                    "current_regime": df["DVI_L"].iloc[-1],
                    "tag": tag
                }
        
        if c_data and m_data and l_data:
            final_json = generate_json_multiperiod(df, c_data, m_data, l_data)
                
            regime_global = get_global_regime(
                    relatorio_c=final_json["relatorio_C"],
                    relatorio_m=final_json["relatorio_M"],
                    relatorio_l=final_json["relatorio_L"],
                    weight_c=0.2,
                    weight_m=0.3,
                    weight_l=0.5
            )
            current_price = float(df["close"].iloc[-1])
            global_risk = get_risk_management_global(regime_global, current_price, risk_management)
                
            final_json["gerenciamento_risco_global"] = {
                    "regime_global": regime_global,
                    "risk_management": global_risk
            }

        # Extraímos a direção e probabilidade do relatório do curto prazo:
        short_dir = final_json["relatorio_C"]["regime_atual"]["direcao"]  # Ex.: "Alta", "Queda", "Neutra"
        short_probs = final_json["relatorio_C"]["probabilidade"]["direcao"]  # Ex.: {"Alta": "5.80%", "Neutra": "88.41%", "Queda": "5.80%"}
        short_prob_value = float(short_probs.get(short_dir, "0.00%").replace("%", "")) / 100.0

        # Extraímos a direção global:
        global_dir = regime_global["direcao_global"]
        # Pegamos os valores atuais de recommended_action e recommended_size
        global_risk_action = final_json["gerenciamento_risco_global"]["risk_management"]["recommended_action"]
        global_risk_size = final_json["gerenciamento_risco_global"]["risk_management"]["recommended_size"]

        # Se o relatório do curto prazo indicar um sinal forte (por exemplo, probabilidade > 50%)
        # e essa direção for diferente do regime global, aplicamos o override
        if short_dir in ["Alta", "Queda"] and short_prob_value > 0.50:
            if short_dir == "Alta" and global_dir != "Alta":
                new_action = "BUY"
                new_size = 0.3  # Exemplo: posição reduzida
                # Definindo stop e take-profit para uma entrada de compra com sinal de curto prazo
            elif short_dir == "Queda" and global_dir != "Queda":
                new_action = "SELL"
                new_size = 0.3
            else:
                new_action = global_risk_action
                new_size = global_risk_size

            final_json["gerenciamento_risco_global"]["risk_management"]["recommended_action"] = new_action
            final_json["gerenciamento_risco_global"]["risk_management"]["recommended_size"] = new_size
                        
        now = datetime.datetime.utcnow()
        date_folder = os.path.join("frontend/json", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
        symbol_folder = os.path.join(date_folder, symbol.replace("/", "-"))
        os.makedirs(symbol_folder, exist_ok=True)
        timestamp_str = now.strftime("%Y%m%dT%H%M%S")
        json_filename = os.path.join(symbol_folder, f"report_HMM_multitimeframe_{symbol.replace('/', '-')}_{TIMEFRAME}_{timestamp_str}.json")
                
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        logger.info(f"Relatório JSON gerado para {symbol}: {json_filename}")

    update_main_json()

# ------------------- EXECUÇÃO DO SCRIPT -------------------
if __name__ == "__main__":
    main()
