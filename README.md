# Wine Quality Classification 🍷

Projeto desenvolvido para o **POSTECH — DTAT — Tech Challenge, Fase 2**. O objetivo é classificar vinhos em duas categorias a partir de suas características físico-químicas.

## Objetivo

Prever a classe de qualidade do vinho com base na coluna `quality`:

- **Alta qualidade (1):** `quality >= 7`
- **Baixa/média qualidade (0):** `quality < 7`

## Estrutura do projeto

```text
wine-quality-classification/
│
├── data/                   # Base de dados utilizada
│   └── WineQT.csv
│
├── notebooks/              # Notebook com a análise e modelagem
│   └── wine_quality_analysis.ipynb
│
├── src/                    # Scripts auxiliares de modelagem
│   └── train_model.py
│
├── results/                # Gráficos e métricas dos modelos
│   └── .gitkeep
│
├── requirements.txt        # Bibliotecas utilizadas
└── README.md               # Descrição do projeto
```

## Como executar

### Opção 1 — Google Colab

1. Abra `notebooks/wine_quality_analysis.ipynb` no Google Colab.
2. Execute as células em ordem.
3. Quando solicitado, carregue o arquivo `WineQT.csv` que está na pasta `data/`.

### Opção 2 — Execução local

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd wine-quality-classification
python -m venv .venv
```

Ative o ambiente virtual:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

Instale as dependências e execute o script:

```bash
pip install -r requirements.txt
python src/train_model.py
```

O script cria os arquivos de métricas e importância das variáveis na pasta `results/`.

## Metodologia

1. Compreensão do problema e criação do alvo binário.
2. Análise exploratória: distribuição, correlação, valores ausentes, duplicatas, outliers e balanceamento das classes.
3. Pré-processamento com imputação da mediana e padronização para a Regressão Logística.
4. Treinamento e comparação de Regressão Logística e Random Forest.
5. Avaliação por acurácia, precisão, recall, F1-score e ROC-AUC.
6. Interpretação da importância das variáveis e discussão das limitações.

## Cuidados na interpretação

As métricas reportadas são referentes à divisão treino/teste usada no experimento e podem variar com mudanças nos dados ou no ambiente. Correlações e importância de variáveis indicam associações usadas pelos modelos, não causalidade. O modelo deve ser validado em dados externos antes de ser utilizado operacionalmente.

## Base de dados

Arquivo utilizado: `data/WineQT.csv`, fornecido para o Tech Challenge.
