

## 💡 EcoAgente: Otimizador de Energia Inteligente (MVP)

[](https://hackathoncid-zb8fhrbwtb8u4dqchepsct.streamlit.app/)


O **EcoAgente** é um Mínimo Produto Viável (MVP) desenvolvido para demonstrar o potencial da **Inteligência Artificial** e **Machine Learning** na otimização de consumo de energia residencial. Atuamos como um agente proativo, analisando dados históricos para prever tendências e alertar o usuário sobre desperdício e picos anômalos.

### 🌟 Funcionalidades e Inovação

| Pilar | Descrição | Tecnologia Chave |
| :--- | :--- | :--- |
| **Detecção de Anomalias** | Identifica picos de consumo que são estatisticamente atípicos (*outliers*) no submedidor de `entretenimento`. O limite é baseado em **1.5 Desvios Padrão** ($\mathbf{1.5\sigma}$) do histórico completo. | Análise Estatística ($\mathbf{\mu + 1.5\sigma}$) |
| **Previsão de Tendência** | Utiliza o consumo histórico para projetar as próximas 24 horas, auxiliando no planejamento e na detecção de desvios futuros. | Regressão Linear (OLS) com **Lag Features** ($\mathbf{t-1}, \mathbf{t-24}$) |
| **Agente Proativo** | Geração de alertas e respostas contextuais amigáveis, usando o poder da IA Generativa para fornecer sugestões de economia. | Google Gemini API (GenAI/RAG) |
| **Análise Interativa** | Permite ao usuário selecionar qualquer dia do histórico para investigação de anomalias (análise diária). | Streamlit |

-----

### 💻 Estrutura do Projeto (Stack)

O projeto foi desenvolvido em Python, utilizando as seguintes bibliotecas principais:

  * **Interface:** `streamlit`, `altair`
  * **ML/Estatística:** `pandas`, `numpy`, `statsmodels` (para OLS)
  * **IA:** `google-genai`

### ▶️ Guia de Execução Local

#### 1\. Pré-requisitos

Certifique-se de ter o Python (3.9+) e as dependências instaladas via `requirements.txt`.

#### 2\. Configuração de Chave API (Segurança)

Para usar o módulo Gemini, você deve configurar sua chave API no arquivo **`.streamlit/secrets.toml`** (que não deve ser versionado no GitHub).

```toml
# Exemplo de conteúdo do secrets.toml
GEMINI_API_KEY = "SUA_CHAVE_API_MUITO_LONGA_E_SECRETA"
```

#### 3\. Dados (IMPORTANTE)

O projeto depende do arquivo de dados `household_power_consumption.txt`. Você deve **copiar este arquivo** para a **raiz** do seu repositório para que o caminho relativo seja encontrado:

```python
# O código espera que o arquivo esteja na raiz do repositório:
RAW_FILE_PATH = "household_power_consumption.txt" 
```

#### 4\. Execução do Aplicativo

```bash
# Instale as dependências (se ainda não o fez)
# pip install -r requirements.txt 

streamlit run seu_app.py
```

-----

### 🎯 Testando a Anomalia

Para validar a funcionalidade de alerta:

1.  Vá para a seção **Detecção de Anomalia e Chat**.
2.  Selecione uma data com pico de consumo conhecido (Ex: **2007-08-11**).
3.  O sistema deve retornar o **ALERTA DE ANOMALIA** com a hora e o valor do pico detectado.
