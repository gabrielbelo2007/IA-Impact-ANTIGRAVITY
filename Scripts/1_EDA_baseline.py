"""
===============================================================================
PROJETO: Impacto de IA nos Estudantes (AI Student Impact Dataset)
ARQUIVO: EDA_baseline.py
OBJETIVO: Análise Exploratória de Dados (EDA) automática, comparação de baselines
          de regressão e inferência no conjunto de teste.
AUTOR: Senior Data Scientist Agent
DATA: 2026-09-17
===============================================================================
"""

# %% [markdown]
# # 1. Importação de Bibliotecas e Configurações

# %%
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Formatação de saída do pandas para melhor legibilidade
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

TRAIN_PATH = "./Database/train.csv"
TEST_PATH = "./Database/test.csv"
PREDICTIONS_OUTPUT_PATH = "./Database/test_predictions.csv"
TARGET_COL = "Skill_Retention_Score"
ID_COL = "Student_ID"
RANDOM_STATE = 42

print("=" * 80)
print("  ANÁLISE EXPLORATÓRIA DE DADOS (EDA) & MODELAGEM DE BASELINE")
print("  Dataset: AI Student Impact Dataset")
print("=" * 80)

# %% [markdown]
# # 2. Carga e Inspeção Inicial dos Dados

# %%
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Arquivo de treino não encontrado em {TRAIN_PATH}")

df_train = pd.read_csv(TRAIN_PATH)
print(f"\n[INFO] Dados de treino carregados com sucesso de: {TRAIN_PATH}")
print(f"       Dimensões: {df_train.shape[0]:,} linhas x {df_train.shape[1]} colunas.")

# Identificação de colunas e tipos
print("\n" + "-" * 80)
print("2.1. ESTRUTURA GERAL DAS COLUNAS E TIPOS DE DADOS")
print("-" * 80)
col_info = pd.DataFrame({
    'Tipo': df_train.dtypes,
    'Não-Nulos': df_train.notnull().sum(),
    'Nulos': df_train.isnull().sum(),
    '% Nulos': (df_train.isnull().sum() / len(df_train)) * 100,
    'Uniques': df_train.nunique()
})
print(col_info)

# %% [markdown]
# # 3. Análise Exploratória Detalhada (EDA)

# %%
print("\n" + "-" * 80)
print("3. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)")
print("-" * 80)

# 3.1. Variável Target
print("\n>>> 3.1. Análise da Variável Alvo (Target):", TARGET_COL)
target_stats = df_train[TARGET_COL].describe()
print(target_stats)
target_skew = df_train[TARGET_COL].skew()
target_kurt = df_train[TARGET_COL].kurtosis()
print(f"Assimetria (Skewness): {target_skew:.4f} (Distribuição aproximadamente simétrica/levemente alongada)")
print(f"Curtose (Kurtosis):     {target_kurt:.4f}")

# 3.2. Análise de Cardinalidade e Distribuição Categórica
print("\n>>> 3.2. Cardinalidade e Distribuição das Variáveis Categóricas/Booleanas")
cat_cols = [c for c in df_train.columns if df_train[c].dtype in ['object', 'bool', 'string'] and c != TARGET_COL]

for c in cat_cols:
    val_counts = df_train[c].value_counts(normalize=True) * 100
    counts_raw = df_train[c].value_counts()
    dist_df = pd.DataFrame({'Total': counts_raw, 'Percentual (%)': val_counts})
    print(f"\nVariável: [{c}] (Cardinalidade = {df_train[c].nunique()}):")
    print(dist_df.to_string())

# 3.3. Análise das Variáveis Numéricas e Correlações
print("\n>>> 3.3. Estatísticas Descritivas das Variáveis Numéricas (Features)")
num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in [ID_COL, TARGET_COL]]
print(df_train[num_cols].describe().T)

print("\n>>> 3.4. Matriz de Correlação Linear de Pearson com o Target:")
corr_with_target = df_train[num_cols + [TARGET_COL]].corr()[TARGET_COL].sort_values(ascending=False)
print(corr_with_target.to_frame(name="Correlação com Target"))

# 3.5. Relação Categórica com o Target
print("\n>>> 3.5. Média e Desvio-Padrão do Target por Categoria Chave:")
for c in ['Prompt_Engineering_Skill', 'Burnout_Risk_Level', 'Institutional_Policy', 'Primary_Use_Case']:
    grouped = df_train.groupby(c)[TARGET_COL].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
    print(f"\nTarget por [{c}]:")
    print(grouped)

# %% [markdown]
# # 4. Considerações Metodológicas: Ideal vs. Baseline

# %%
print("\n" + "=" * 80)
print("4. CONSIDERAÇÕES DE ENGENHARIA DE DADOS & MODELAGEM")
print("=" * 80)
print("""
[PROCEDIMENTOS IDEAIS PARA AMBIENTE DE PRODUÇÃO / SOLUÇÃO DEFINITIVA]:
1. Feature Engineering Avançada:
   - Razão de horas de estudo: Weekly_GenAI_Hours / (Traditional_Study_Hours + 1).
   - Variação do GPA: Delta_GPA = Post_Semester_GPA - Pre_Semester_GPA.
   - Índice de Dependência x Eficiência: (Perceived_AI_Dependency * Weekly_GenAI_Hours).
2. Tratamento Específico de Cardinalidade e Ordinalidade:
   - Ordinal Encoding ordenado logicamente para:
     * Year_of_Study (Freshman < Sophomore < Junior < Senior < Graduate).
     * Prompt_Engineering_Skill (Beginner < Intermediate < Advanced).
     * Burnout_Risk_Level (Low < Medium < High).
   - Target Encoding ou One-Hot com regularização para categorias nominais (Major_Category, Primary_Use_Case).
3. Normalização e Robustez:
   - RobustScaler ou QuantileTransformer para variáveis assimétricas e com outliers.
4. Estratégia de Validação:
   - K-Fold Cross-Validation Estratificado por quantis do Target (10 Folds).
5. Algoritmos Avançados:
   - Gradient Boosting (LightGBM, XGBoost, CatBoost) com busca Bayesiana de hiperparâmetros.

[PROCEDIMENTOS APLICADOS NO BASELINE (SIMPLES, ROBUSTO E REPRODUTÍVEL)]:
1. Exclusão do Student_ID (evita data leakage e memorização espúria de ID).
2. Imputação rápida de valores ausentes (Mediana para numéricos, Moda para categóricos).
3. One-Hot Encoding para todas as variáveis categóricas (lidando com novas categorias via ignore).
4. Divisão Holdout simples de 80% treino e 20% validação (random_state fixo).
5. Treinamento comparativo de 4 modelos de baseline com diferentes níveis de complexidade:
   - Baseline 0: DummyRegressor (Previsão da média - Benchmark zero-skill)
   - Baseline 1: Regressão Linear (OLS - Relações lineares diretas)
   - Baseline 2: Ridge Regression (Regressão Linear com Regularização L2)
   - Baseline 3: Random Forest Regressor (Ensemble não-linear com árvores de decisão)
""")

# %% [markdown]
# # 5. Pré-processamento e Divisão de Dados

# %%
print("-" * 80)
print("5. PRÉ-PROCESSAMENTO E DIVISÃO TREINO / VALIDAÇÃO (80 / 20)")
print("-" * 80)

# Separação de X e y
features = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
X = df_train[features].copy()
y = df_train[TARGET_COL].copy()

# Identificação dos tipos de colunas para o pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool', 'string']).columns.tolist()

print(f"[Features Numéricas]   ({len(numeric_features)}): {numeric_features}")
print(f"[Features Categóricas] ({len(categorical_features)}): {categorical_features}")

# Pipelines de pré-processamento via ColumnTransformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Holdout 80% Treino e 20% Validação
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"\nDivisão realizada com sucesso:")
print(f"  Treino:     {X_train.shape[0]:,} amostras ({(len(X_train)/len(X))*100:.1f}%)")
print(f"  Validação:  {X_val.shape[0]:,} amostras ({(len(X_val)/len(X))*100:.1f}%)")

# %% [markdown]
# # 6. Treinamento e Avaliação dos Modelos de Baseline

# %%
print("\n" + "=" * 80)
print("6. TREINAMENTO DOS MODELOS DE BASELINE")
print("=" * 80)

# Dicionário de modelos a avaliar
models = {
    "DummyRegressor (Média)": DummyRegressor(strategy="mean"),
    "Regressão Linear (OLS)": LinearRegression(),
    "Ridge Regression (L2)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
}

results_list = []
trained_pipelines = {}

for name, model in models.items():
    print(f"\n--> Treinando [{name}]...")
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Ajuste no conjunto de treino
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    # Avaliação no treino e na validação
    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    
    # Cálculo das métricas de validação
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val) * 100
    
    # Métricas de treino para checar sobreajuste (overfitting)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    results_list.append({
        "Modelo": name,
        "MSE (Val)": mse_val,
        "RMSE (Val)": rmse_val,
        "MAE (Val)": mae_val,
        "R² (Val)": r2_val,
        "MAPE (%)": mape_val,
        "MSE (Treino)": mse_train,
        "RMSE (Treino)": rmse_train,
        "R² (Treino)": r2_train
    })

# %% [markdown]
# # 7. Ranking dos Modelos e Seleção do Campeão

# %%
print("\n" + "=" * 80)
print("7. RANKING E COMPARATIVO DE PERFORMANCE NA VALIDAÇÃO")
print("=" * 80)

df_results = pd.DataFrame(results_list)
# Ordenar por menor MSE na validação (e maior R²)
df_results = df_results.sort_values(by="MSE (Val)", ascending=True).reset_index(drop=True)
df_results.index = df_results.index + 1
df_results.index.name = "Posição"

print(df_results.to_string())

# Seleção do melhor modelo
best_model_name = df_results.iloc[0]["Modelo"]
best_model_mse = df_results.iloc[0]["MSE (Val)"]
best_model_rmse = df_results.iloc[0]["RMSE (Val)"]
best_model_r2 = df_results.iloc[0]["R² (Val)"]
best_pipeline = trained_pipelines[best_model_name]

print("\n" + "*" * 80)
print(f"  MODELO VENCEDOR SELECIONADO: [{best_model_name}]")
print(f"  MSE de Validação:  {best_model_mse:.4f}")
print(f"  RMSE de Validação: {best_model_rmse:.4f}")
print(f"  R² de Validação:   {best_model_r2:.4f}")
print("*" * 80)

# %% [markdown]
# # 8. Inferência Única no Conjunto de Teste (./Database/test.csv)

# %%
print("\n" + "=" * 80)
print("8. EXECUÇÃO DA INFERÊNCIA NO CONJUNTO DE TESTE")
print("=" * 80)

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Arquivo de teste não encontrado em {TEST_PATH}")

df_test = pd.read_csv(TEST_PATH)
print(f"[INFO] Dataset de teste carregado de: {TEST_PATH}")
print(f"       Dimensões: {df_test.shape[0]:,} linhas x {df_test.shape[1]} colunas.")

# Verificar ausência do target no teste
has_target_in_test = TARGET_COL in df_test.columns
print(f"       Presença da coluna Target '{TARGET_COL}' no teste: {has_target_in_test}")

# Garantir mesmas colunas de entrada
X_test = df_test[features].copy()

# Inferência ÚNICA com o melhor modelo
print(f"\n[INFERÊNCIA] Executando predições com o modelo vencedor [{best_model_name}]...")
test_predictions = best_pipeline.predict(X_test)

# Criação do DataFrame de submissão/predições
df_predictions = pd.DataFrame({
    ID_COL: df_test[ID_COL],
    f"{TARGET_COL}_Pred": np.clip(test_predictions, 0.0, 100.0) # Pontuação no intervalo válido [0, 100]
})

# Resumo Estatístico das Predições
print("\n" + "-" * 80)
print("ESTATÍSTICAS DESCRITIVAS DAS PREDIÇÕES GERADAS NO CONJUNTO DE TESTE:")
print("-" * 80)
print(df_predictions[f"{TARGET_COL}_Pred"].describe().to_frame(name="Estatísticas Predição").to_string())

# Salvar predições em arquivo CSV
df_predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
print(f"\n[SUCESSO] Predições salvas em: {PREDICTIONS_OUTPUT_PATH}")

# Amostra das predições
print("\nPrimeiras 10 linhas das predições geradas:")
print(df_predictions.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
print("=" * 80)
