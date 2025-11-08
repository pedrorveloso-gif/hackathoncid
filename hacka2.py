import streamlit as st
import pandas as pd
import numpy as np
import os 
from google import genai 
from google.genai import types 
from datetime import timedelta, date 
from statsmodels.formula.api import ols 
import altair as alt

# --- 0. CONFIGURAÇÃO E INICIALIZAÇÃO ROBUSTA DA API ---
api_status = False
client = None
GEMINI_MODEL = 'gemini-2.5-flash' 

try:
    API_KEY = st.secrets["GEMINI_API_KEY"] 
    os.environ["GEMINI_API_KEY"] = API_KEY
    client = genai.Client() 
    api_status = True
except KeyError:
    st.error("Erro: Chave 'GEMINI_API_KEY' não encontrada. Verifique se o arquivo .streamlit/secrets.toml está correto e nomeado como 'GEMINI_API_KEY'.")
    client = None
except Exception as e:
    st.error(f"Erro ao inicializar a API do Gemini: {e}")
    client = None
    
# --- ARQUIVOS DE DADOS ---
# ATENÇÃO: Verifique se este caminho é o correto no seu ambiente!
RAW_FILE_PATH = data/household_power_consumption.txt

# --- FUNÇÕES DE ML E ETL ---
@st.cache_data(show_spinner="Carregando e limpando dados brutos...")
def load_and_prepare_raw_data(file_path):
    """Carrega e limpa o DataFrame BRUTO."""
    df = pd.read_csv(
        file_path, 
        sep=';', 
        low_memory=False, 
        na_values=['?'],
        parse_dates={'Datetime': ['Date', 'Time']}, 
        infer_datetime_format=True
    )
    df.rename(columns={
        'Global_active_power': 'consumo_energia',
        'Sub_metering_1': 'cozinha', 'Sub_metering_2': 'lavanderia', 'Sub_metering_3': 'entretenimento'
    }, inplace=True)
    numericos_foco = ['consumo_energia', 'cozinha', 'lavanderia', 'entretenimento']
    for col in numericos_foco:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['consumo_energia'], inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[numericos_foco] 

def calculate_hourly_average(df_full):
    """Calcula a média de consumo por hora do dia em todo o histórico."""
    df_full['Hour'] = df_full.index.hour
    df_hourly_avg = df_full.groupby('Hour')['consumo_energia'].mean().reset_index()
    df_hourly_avg.rename(columns={'consumo_energia': 'Média de Consumo (kW)'}, inplace=True) 
    df_hourly_avg['Hora'] = df_hourly_avg['Hour'].apply(lambda x: f'{x:02d}:00')
    return df_hourly_avg[['Hora', 'Média de Consumo (kW)']]

@st.cache_data(show_spinner="Calculando métricas e treinando ML (Regressão Simples com Lag)...")
def calculate_ml_and_metrics(df_full):
    """Calcula todas as métricas ML/Estatísticas no df_full."""
    
    # 1. MÉTRICAS DE ANOMALIA (Threshold)
    df_madrugada = df_full.iloc[:45000][(df_full.iloc[:45000].index.hour >= 2) & (df_full.iloc[:45000].index.hour <= 5)]
    # 💡 CORREÇÃO/AJUSTE DE SENSIBILIDADE: Usando 2.0 desvios padrão (era 3.0) para detectar anomalias mais facilmente.
    threshold_anomalia = df_madrugada['entretenimento'].mean() + (2.0 * df_madrugada['entretenimento'].std())
    
    # 2. TREINAMENTO E PREVISÃO ML
    df_hist_ml = df_full['consumo_energia'].resample('H').mean().dropna().to_frame()
    df_previsao_plot = pd.DataFrame(columns=['Hora', 'Previsao'])
    r_squared = "0.0000"
    
    if len(df_hist_ml) >= 48: 
        try:
            df_hist_ml['Hour'] = df_hist_ml.index.hour
            df_hist_ml['consumo_lag_1h'] = df_hist_ml['consumo_energia'].shift(1)
            df_hist_ml['consumo_lag_24h'] = df_hist_ml['consumo_energia'].shift(24)
            df_hist_ml.dropna(inplace=True)
            
            model = ols('consumo_energia ~ C(Hour) + consumo_lag_1h + consumo_lag_24h', data=df_hist_ml).fit()
            r_squared = f"{model.rsquared:.4f}"
            
            future_hours = pd.DataFrame({'Hour': range(24)})
            initial_lag_1h = df_hist_ml['consumo_energia'].iloc[-1]
            initial_lag_24h = df_hist_ml['consumo_energia'].iloc[-24] 

            forecasts = []
            current_lag_1h = initial_lag_1h 
            
            for i in range(24):
                prediction_input = pd.DataFrame({
                    'Hour': [i], 
                    'consumo_lag_1h': [current_lag_1h],
                    'consumo_lag_24h': [initial_lag_24h] 
                })
                
                forecast = model.predict(prediction_input)[0]
                forecasts.append(forecast)
                current_lag_1h = forecast 
                
            df_previsao_plot['Hora'] = future_hours['Hour'].apply(lambda x: f'{x:02d}:00')
            df_previsao_plot['Previsao'] = np.array(forecasts).round(3)

        except Exception as e:
            pass 
            
    return threshold_anomalia, df_previsao_plot, r_squared

# --- PASSO DE EXECUÇÃO (Cálculo de tudo) ---
df_completo_snapshot = load_and_prepare_raw_data(RAW_FILE_PATH)
threshold_anomalia, df_previsao_plot, r_squared_score = calculate_ml_and_metrics(df_completo_snapshot)

# Agregação para Plotagem
df_historico_anual_plot = df_completo_snapshot['consumo_energia'].resample('Y').mean().reset_index()
df_historico_anual_plot.rename(columns={'consumo_energia': 'Média de Consumo Anual (kW)'}, inplace=True)
df_historico_anual_plot['Ano'] = df_historico_anual_plot['Datetime'].dt.year.astype(str)
df_consumo_por_hora = calculate_hourly_average(df_completo_snapshot)

# --- FUNÇÕES DE ANOMALIA E GEMINI ---

def check_for_anomaly(df_day, threshold):
    """
    Verifica se houve QUALQUER registro anômalo (entretenimento > threshold) 
    em algum minuto do dia selecionado (df_day).
    """
    if df_day.empty: 
        return {"status": False, "msg": "DataFrame do dia vazio."}

    # 1. Checagem da Anomalia
    anomalous_points = df_day[df_day['entretenimento'] > threshold]
    
    if not anomalous_points.empty:
        # Se encontrou, pega o ponto de maior pico para exibição
        peak_anomaly = anomalous_points.sort_values(by='entretenimento', ascending=False).iloc[0]
        
        return {
            "status": True,
            "consumo_atual": peak_anomaly['entretenimento'],
            "threshold": threshold,
            "horario_completo": peak_anomaly.name,
            "horario_atual": peak_anomaly.name.strftime("%H:%M"),
            "ativo": "TV/Entretenimento"
        }
    
    # Se não houver anomalia, retorna False usando o último timestamp do dia para referência
    return {"status": False, "horario_completo": df_day.index[-1]}

def _run_gemini_query(system_instruction, user_content):
    if not client: return "Desculpe, o Agente está offline (API Key)."
    try:
        contents = [types.Content(parts=[types.Part(text=user_content)])]
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )
        )
        return response.text
    except Exception as e:
        return f"Erro na API ao gerar resposta (Gemini): {e}"

def generate_agent_response(anomalia_payload=None, user_query=None):
    if anomalia_payload and anomalia_payload.get("status"):
        system_prompt = "Você é o EcoAgente, focado em alertas proativos."
        context = (
            f"Alerta: Consumo anômalo no {anomalia_payload['ativo']} às {anomalia_payload['horario_atual']}. "
            f"Pico: {anomalia_payload['consumo_atual']:.3f}kW. Threshold: {anomalia_payload['threshold']:.3f}kW. "
            "Sugira uma ação imediata de automação."
        )
        user_prompt_final = "Crie um alerta amigável e proativo."
        return _run_gemini_query(system_prompt, f"Contexto: {context}. Instrução: {user_prompt_final}")
    else:
        system_prompt = "Você é o EcoAgente. Responda brevemente e amigavelmente. Seu foco é a economia de energia."
        user_prompt_final = user_query if user_query else "Gere uma mensagem de boas-vindas."
        return _run_gemini_query(system_prompt, user_prompt_final)

def _get_rag_response_with_data(user_query, context_data):
    if not client: return f"Desculpe, a IA está offline, mas o valor é: {context_data}"
    system_prompt = "Você é o EcoAgente. Use o contexto de dados fornecido para responder à pergunta do usuário de forma direta e amigável. Não adicione informações externas e forneça o valor numérico exato."
    user_content = f"Contexto de Dados: {context_data}. Pergunta do usuário: {user_query}"
    return _run_gemini_query(system_prompt, user_content)


# --- 5. INTERFACE STREAMLIT ---
st.set_page_config(layout="wide", page_title="EcoAgente MVP Estático")
st.title("💡 EcoAgente: Otimizador de Energia")

# 5.1. Contexto Histórico Anual
st.header("Contexto Histórico (Médias Anuais)")
chart_anual = alt.Chart(df_historico_anual_plot).mark_bar().encode(
    x=alt.X('Ano:N', axis=alt.Axis(title='Ano')), 
    y=alt.Y('Média de Consumo Anual (kW):Q', title='Média de Consumo (kW)'), 
    tooltip=['Ano', alt.Tooltip('Média de Consumo Anual (kW)', format='.3f')]
).properties(
    title='Média Anual de Consumo de Energia',
    height=300
).interactive()
st.altair_chart(chart_anual, use_container_width=True)
st.markdown("---")


# 5.2. Previsão ML
st.header("Previsão: Próximas 24 Horas")
st.caption(f"Previsão de consumo por hora, gerada por um modelo de Regressão Linear **Aprimorada** (Lag). **Coerência (R²): {r_squared_score}**.")

if not df_previsao_plot.empty:
    chart_previsao_base = alt.Chart(df_previsao_plot).encode(
        x=alt.X('Hora:O', sort=None, title='Hora do Dia', axis=alt.Axis(
            labelAngle=-45, 
            values=[f'{h:02d}:00' for h in range(0, 24, 3)] 
        )),
        y=alt.Y('Previsao:Q', title='Consumo Previsto (kW)')
    ).properties(
        title='Previsão de Consumo de Energia (Próximas 24h)',
        height=300
    ).interactive()
    
    chart_line = chart_previsao_base.mark_line(
        color='#1E90FF', 
        strokeWidth=3 
    ).encode(
        tooltip=['Hora', alt.Tooltip('Previsao', format='.3f')]
    )

    chart_points = chart_previsao_base.mark_circle(size=60).encode(
        color=alt.value('#ADD8E6'), 
        tooltip=['Hora', alt.Tooltip('Previsao', format='.3f')]
    )
    
    st.altair_chart(chart_line + chart_points, use_container_width=True)

else:
    st.warning("Não foi possível gerar a previsão para as próximas 24 horas (O modelo ML falhou).")

st.markdown("---")

# 5.3. Gráfico de Desempenho Típico
st.header("Desempenho Típico: Média de Consumo por Hora do Dia")
st.caption("Compara o seu consumo em qualquer dia com o padrão histórico (2006-2010).")

chart_consumo_hora = alt.Chart(df_consumo_por_hora).mark_bar().encode(
    x=alt.X('Hora:O', sort=None, axis=alt.Axis(title='Hora do Dia')), 
    y=alt.Y('Média de Consumo (kW):Q', title='Média de Consumo (kW)'),
    tooltip=['Hora', alt.Tooltip('Média de Consumo (kW)', format='.3f')]
).properties(
    title='Média Histórica de Consumo por Hora do Dia',
    height=300
).interactive()
st.altair_chart(chart_consumo_hora, use_container_width=True)

st.markdown(f"**Threshold de Standby (Madrugada):** {threshold_anomalia:.3f} kW")


# --- SEÇÃO DE DETECÇÃO DE ANOMALIA E CHAT (ANÁLISE DIÁRIA) ---
st.markdown("---")
st.header("Detecção de Anomalia e Chat")

# 1. Definição do Range de Datas
min_date = df_completo_snapshot.index.min().date()
max_date = df_completo_snapshot.index.max().date()
default_date = date(2007, 2, 2) # Sugerindo 2007-02-02 para testar a anomalia

col_alert, col_chat = st.columns([1, 1])

with col_alert:
    st.subheader("Análise de Consumo Diário") 
    
    # 2. Seletor de Data (Dia apenas)
    selected_date = st.date_input(
        "Selecione o Dia para Análise", 
        # Define 2007-02-02 como valor padrão para demonstrar a anomalia
        value=default_date if min_date <= default_date <= max_date else min_date,
        min_value=min_date, 
        max_value=max_date,
        key="date_selector"
    )
    
    # 3. Encontra todos os dados para o dia selecionado (df_day)
    df_day = df_completo_snapshot[df_completo_snapshot.index.date == selected_date]
    
    if not df_day.empty:
        
        # 4. Realiza a Checagem de Anomalia
        anomalia_data = check_for_anomaly(df_day, threshold_anomalia)
        
        # 5. Define a hora de exibição 
        current_time_display = anomalia_data['horario_completo']
        
        # 6. Exibe o Resultado
        st.subheader(f"Resultado da Análise do Dia {current_time_display.strftime('%Y-%m-%d')}")

        if anomalia_data["status"]:
            st.error(f"🚨 CONSUMO MAIS ALTO DO DIA DETECTADO ÀS **{anomalia_data['horario_atual']}**.")
            st.markdown(f"**Ativo Suspeito:** {anomalia_data['ativo']}")
            st.markdown(f"**Pico Registrado:** {anomalia_data['consumo_atual']:.3f}kW") 
            
            agent_message = generate_agent_response(anomalia_payload=anomalia_data)
            st.markdown("---")
            st.info(agent_message)
        else:
            st.success("✅ Consumo em Faixa Normal ao longo do dia.")
            agent_message = generate_agent_response(anomalia_payload={"status": False})
            st.info(agent_message)
            
    else:
        st.warning("Não há dados disponíveis para o dia selecionado. Por favor, escolha outra data.")
        agent_message = generate_agent_response(anomalia_payload={"status": False})
        st.info(agent_message)


with col_chat:
    st.subheader("Voz do EcoAgente")
    
    user_query = st.text_input("Sua pergunta:", key="chat_input")
    
    if user_query:
        st.info(f"Você: {user_query}")
        agent_response = None
        
        if 'média de consumo no ano de' in user_query.lower() or 'consumo médio anual' in user_query.lower():
            found_year = next((s for s in user_query.split() if s.isdigit() and len(s) == 4), None)
            
            if found_year:
                try:
                    target_year = int(found_year)
                    df_year = df_completo_snapshot[df_completo_snapshot.index.year == target_year]
                    if not df_year.empty:
                        avg_consumption = df_year['consumo_energia'].mean()
                        context_data = f"A média anual de consumo no ano {target_year} é de {avg_consumption:.3f} kW."
                        agent_response = _get_rag_response_with_data(user_query, context_data)
                    else:
                        agent_response = f"Não encontrei dados para o ano {target_year} no meu histórico. Lembre-se, meu histórico vai de 2006 a 2010."
                except Exception as e:
                    agent_response = f"Desculpe, houve um erro interno ao calcular essa média. Erro: {e}"
        
        elif 'previsão para 24 horas' in user_query.lower() or 'próximas 24 horas' in user_query.lower():
            
            if not df_previsao_plot.empty:
                max_pred = df_previsao_plot['Previsao'].max()
                min_pred = df_previsao_plot['Previsao'].min()
                context_data = f"A previsão para as próximas 24 horas varia entre {min_pred:.3f} kW (noite) e {max_pred:.3f} kW (pico do dia). O modelo tem um R² de {r_squared_score}."
                agent_response = _get_rag_response_with_data(user_query, context_data)
            else:
                agent_response = "Não foi possível gerar a previsão de consumo para as próximas 24 horas."
            
        if agent_response is None:
            agent_response = generate_agent_response(user_query=user_query)


        st.success(f"EcoAgente: {agent_response}")
