"""
===============================================================================
PROJETO: Impacto de IA nos Estudantes (AI Student Impact Dataset)
ARQUIVO: model_optimization.py
OBJETIVO: Pipeline avançado de modelagem preditiva para Skill_Retention_Score.
          - Avaliação das considerações metodológicas da Etapa 4 da EDA
          - Engenharia de Features avançada e Encoders semânticos (Ordinais/Nominais)
          - Prevenção rigorosa de Data Leakage e Overfitting
          - Validação Cruzada 5-Fold Reprodutível
          - Exploração exaustiva de modelos baseados em árvore e boosting
          - Blending/Ensemble ponderado out-of-fold
          - Seleção do pipeline campeão pelo menor MSE de validação
          - Inferência final única no conjunto de teste (Database/test.csv)
AUTOR: Senior Data Scientist Agent
DATA: 2026-09-17
===============================================================================
"""

# %% [markdown]
# # 1. Importação de Módulos e Configuração do Ambiente

# %%
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Modelos baseados em árvore e boosting
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor
)

warnings.filterwarnings("ignore")

# Configurações de exibição do Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

TRAIN_PATH = "./Database/train.csv"
TEST_PATH = "./Database/test.csv"
PREDICTIONS_OUTPUT_PATH = "./Database/test_predictions.csv"
PREDICTIONS_OPT_PATH = "./Database/test_predictions_optimized.csv"
TARGET_COL = "Skill_Retention_Score"
ID_COL = "Student_ID"
RANDOM_STATE = 42
N_SPLITS = 5

print("=" * 85)
print("   OTIMIZAÇÃO AVANÇADA DE PIPELINE PREDITIVO - SKILL RETENTION SCORE")
print("=" * 85)

# %% [markdown]
# # 2. Avaliação das Considerações Metodológicas (Etapa 4 do EDA)

# %%
print("\n" + "=" * 85)
print("2. AVALIAÇÃO DAS DIRETRIZES METODOLÓGICAS (ANTI-LEAKAGE E ENGENHARIA)")
print("=" * 85)
print("""
[DIRETRIZES DE ENGENHARIA DE ATRIBUTOS IMPLEMENTADAS]:
1. GPA_Delta: Post_Semester_GPA - Pre_Semester_GPA (Diferença de rendimento acadêmico).
2. Total_Study_Hours: Traditional_Study_Hours + Weekly_GenAI_Hours (Carga horária total).
3. Study_Ratio: Weekly_GenAI_Hours / (Traditional_Study_Hours + 1.0) (Intensidade de IA vs Estudo Tradicional).
4. Traditional_Study_Proportion: Traditional_Study_Hours / (Total_Study_Hours + 1e-5).
5. AI_Intensity_Index: Perceived_AI_Dependency * Weekly_GenAI_Hours (Sobrecarga de dependência de IA).
6. Exam_Stress_Impact: Anxiety_Level_During_Exams / (Pre_Semester_GPA + 0.1).
7. Effective_AI_Usage: Prompt_Engineering_Skill (ordinal) * Weekly_GenAI_Hours.
8. Tool_Efficiency: Tool_Diversity / (Weekly_GenAI_Hours + 1.0).

[TRATAMENTO SEMÂNTICO DE VARIÁVEIS]:
- Ordinal Encoding com ordem semântica real preservada:
  * Year_of_Study: Freshman (1) < Sophomore (2) < Junior (3) < Senior (4) < Graduate (5)
  * Prompt_Engineering_Skill: Beginner (1) < Intermediate (2) < Advanced (3)
  * Burnout_Risk_Level: Low (1) < Medium (2) < High (3)
  * Institutional_Policy: Strict_Ban (1) < Allowed_With_Citation (2) < Actively_Encouraged (3)
  * Paid_Subscription: False (0), True (1)
- One-Hot Encoding para categorias nominais (Major_Category, Primary_Use_Case).

[BLINDAGEM RIGOROSA CONTRA OVERFITTING E DATA LEAKAGE]:
- O conjunto de teste (./Database/test.csv) permanece INTOCADO durante toda a fase de treinamento,
  validação cruzada e seleção de hiperparâmetros.
- Validação Cruzada K-Fold (5 Folds): Todas as transformações e ajustes ocorrem estritamente
  dentro do fold de treino e são aplicadas no fold de validação out-of-fold (OOF).
- Monitoramento contínuo da razão MSE_Treino / MSE_Val para detectar sobreajuste.
""")

# %% [markdown]
# # 3. Carga de Dados e Transformador Customizado de Features

# %%
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Arquivo de treino não encontrado em {TRAIN_PATH}")

df_train = pd.read_csv(TRAIN_PATH)
print(f"[INFO] Dados de treino carregados com sucesso: {df_train.shape[0]:,} linhas x {df_train.shape[1]} colunas.")

# Mapeamentos ordinais explícitos
ORDINAL_MAPPINGS = {
    'Year_of_Study': {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5},
    'Prompt_Engineering_Skill': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3},
    'Burnout_Risk_Level': {'Low': 1, 'Medium': 2, 'High': 3},
    'Institutional_Policy': {'Strict_Ban': 1, 'Allowed_With_Citation': 2, 'Actively_Encouraged': 3},
    'Paid_Subscription': {False: 0, True: 1}
}

class AdvancedFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformador de Engenharia de Features para enriquecimento dos dados.
    Garante reprodutibilidade e encapsulamento em Pipeline anti-leakage.
    """
    def __init__(self, apply_engineering=True):
        self.apply_engineering = apply_engineering
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Aplicação de Mapeamentos Ordinais
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in X_out.columns:
                X_out[col + '_Ord'] = X_out[col].map(mapping).fillna(0).astype(float)
                
        if self.apply_engineering:
            # 2. Engenharia de Relações de Estudo e Desempenho
            X_out['GPA_Delta'] = X_out['Post_Semester_GPA'] - X_out['Pre_Semester_GPA']
            X_out['Total_Study_Hours'] = X_out['Traditional_Study_Hours'] + X_out['Weekly_GenAI_Hours']
            X_out['Study_Ratio'] = X_out['Weekly_GenAI_Hours'] / (X_out['Traditional_Study_Hours'] + 1.0)
            X_out['Traditional_Study_Prop'] = X_out['Traditional_Study_Hours'] / (X_out['Total_Study_Hours'] + 1e-5)
            
            # 3. Interações de IA e Comportamento
            X_out['AI_Intensity_Index'] = X_out['Perceived_AI_Dependency'] * X_out['Weekly_GenAI_Hours']
            X_out['Exam_Stress_Impact'] = X_out['Anxiety_Level_During_Exams'] / (X_out['Pre_Semester_GPA'] + 0.1)
            X_out['Effective_AI_Usage'] = X_out['Prompt_Engineering_Skill_Ord'] * X_out['Weekly_GenAI_Hours']
            X_out['Tool_Efficiency'] = X_out['Tool_Diversity'] / (X_out['Weekly_GenAI_Hours'] + 1.0)
            
        return X_out

# %% [markdown]
# # 4. Definição do Pré-processamento e Pipelines

# %%
def build_preprocessor(apply_fe=True):
    # Colunas nominais a serem codificadas com One-Hot
    nominal_cols = ['Major_Category', 'Primary_Use_Case']
    
    # Colunas numéricas base
    base_numeric = [
        'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity',
        'Traditional_Study_Hours', 'Perceived_AI_Dependency',
        'Anxiety_Level_During_Exams', 'Post_Semester_GPA'
    ]
    
    # Colunas ordinais transformadas em numéricas
    ord_numeric = [c + '_Ord' for c in ORDINAL_MAPPINGS.keys()]
    
    # Colunas geradas pela engenharia
    fe_numeric = [
        'GPA_Delta', 'Total_Study_Hours', 'Study_Ratio',
        'Traditional_Study_Prop', 'AI_Intensity_Index',
        'Exam_Stress_Impact', 'Effective_AI_Usage', 'Tool_Efficiency'
    ] if apply_fe else []
    
    numeric_features = base_numeric + ord_numeric + fe_numeric
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, nominal_cols)
        ]
    )
    
    return Pipeline([
        ('fe', AdvancedFeatureEngineering(apply_engineering=apply_fe)),
        ('preprocessor', col_transformer)
    ])

# %% [markdown]
# # 5. Protocolo de Validação Cruzada 5-Fold Anti-Leakage

# %%
X = df_train.drop(columns=[ID_COL, TARGET_COL]).copy()
y = df_train[TARGET_COL].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Definir catálogo de candidatos para exploração exaustiva
candidate_models = {
    # 1. Baseline RF de referência (mesmos hiperparâmetros do baseline simples)
    "1. Baseline RF (Features Originais)": {
        "apply_fe": False,
        "model": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    },
    
    # 2. Random Forest com Engenharia de Atributos & Ordinais
    "2. Random Forest Otimizado + FE": {
        "apply_fe": True,
        "model": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=15,
            max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1
        )
    },
    
    # 3. ExtraTreesRegressor (Árvores extremamente aleatórias para redução de variância)
    "3. Extra Trees Regressor + FE": {
        "apply_fe": True,
        "model": ExtraTreesRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=15,
            max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1
        )
    },
    
    # 4. HistGradientBoostingRegressor (Gradient Boosting rápido com regularização L2 e early stopping)
    "4. HistGradientBoosting + FE": {
        "apply_fe": True,
        "model": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=3.0, random_state=RANDOM_STATE
        )
    },
    
    # 5. GradientBoostingRegressor (Gradient Boosting clássico com subsampling e regularização)
    "5. Gradient Boosting (GBR) + FE": {
        "apply_fe": True,
        "model": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=5,
            subsample=0.85, min_samples_leaf=20, random_state=RANDOM_STATE
        )
    }
}

print(f"\n[CONFIGURAÇÃO DA VALIDAÇÃO]:")
print(f"  Total de Amostras de Treino: {len(X):,}")
print(f"  Folds de Validação Cruzada:  {N_SPLITS}")
print(f"  Random State Fixado:         {RANDOM_STATE}")
print(f"  Candidatos a Explorar:       {len(candidate_models)}")

# %% [markdown]
# # 6. Execução da Validação Cruzada e Coleta Out-of-Fold (OOF)

# %%
oof_predictions = {name: np.zeros(len(y)) for name in candidate_models.keys()}
cv_summary = []

for model_name, config in candidate_models.items():
    print(f"\n" + "-" * 85)
    print(f"--> Executando 5-Fold CV: [{model_name}]")
    print("-" * 85)
    
    val_mse_folds = []
    val_rmse_folds = []
    val_mae_folds = []
    val_r2_folds = []
    val_mape_folds = []
    train_mse_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val_fold = X.iloc[val_idx], y[val_idx]
        
        # Pipeline independente por fold: blindagem total contra leakage
        pipeline = Pipeline([
            ('prep', build_preprocessor(apply_fe=config['apply_fe'])),
            ('reg', config['model'])
        ])
        
        pipeline.fit(X_tr, y_tr)
        
        # Previsão out-of-fold
        pred_val = pipeline.predict(X_va)
        oof_predictions[model_name][val_idx] = pred_val
        
        # Previsão de treino (para checagem de overfitting)
        pred_train = pipeline.predict(X_tr)
        
        # Métricas do fold
        m_val_mse = mean_squared_error(y_val_fold, pred_val)
        m_val_rmse = np.sqrt(m_val_mse)
        m_val_mae = mean_absolute_error(y_val_fold, pred_val)
        m_val_r2 = r2_score(y_val_fold, pred_val)
        m_val_mape = mean_absolute_percentage_error(y_val_fold, pred_val) * 100
        m_tr_mse = mean_squared_error(y_tr, pred_train)
        
        val_mse_folds.append(m_val_mse)
        val_rmse_folds.append(m_val_rmse)
        val_mae_folds.append(m_val_mae)
        val_r2_folds.append(m_val_r2)
        val_mape_folds.append(m_val_mape)
        train_mse_folds.append(m_tr_mse)
        
    mean_val_mse = np.mean(val_mse_folds)
    mean_val_rmse = np.mean(val_rmse_folds)
    mean_val_mae = np.mean(val_mae_folds)
    mean_val_r2 = np.mean(val_r2_folds)
    mean_val_mape = np.mean(val_mape_folds)
    mean_train_mse = np.mean(train_mse_folds)
    overfitting_ratio = mean_val_mse / (mean_train_mse + 1e-5)
    
    print(f"    [Resultado Médio 5-Fold]")
    print(f"    MSE Val:  {mean_val_mse:.4f}  (± {np.std(val_mse_folds):.4f})")
    print(f"    RMSE Val: {mean_val_rmse:.4f}")
    print(f"    MAE Val:  {mean_val_mae:.4f}")
    print(f"    R² Val:   {mean_val_r2:.4f}")
    print(f"    MAPE Val: {mean_val_mape:.2f}%")
    print(f"    MSE Trn:  {mean_train_mse:.4f} | Razão Overfit (Val/Trn): {overfitting_ratio:.2f}")
    
    cv_summary.append({
        "Modelo": model_name,
        "MSE (Val)": mean_val_mse,
        "RMSE (Val)": mean_val_rmse,
        "MAE (Val)": mean_val_mae,
        "R² (Val)": mean_val_r2,
        "MAPE (%)": mean_val_mape,
        "MSE (Treino)": mean_train_mse,
        "Razão Overfit": overfitting_ratio
    })

# %% [markdown]
# # 7. Otimização de Ensemble Ponderado (Blending Out-of-Fold)

# %%
print("\n" + "=" * 85)
print("7. OTIMIZAÇÃO DE ENSEMBLE PONDERADO (BLENDING OUT-OF-FOLD)")
print("=" * 85)

# Seleção dos modelos avançados com FE para ensemble
models_for_blend = [
    "2. Random Forest Otimizado + FE",
    "3. Extra Trees Regressor + FE",
    "4. HistGradientBoosting + FE",
    "5. Gradient Boosting (GBR) + FE"
]

blend_matrix = np.column_stack([oof_predictions[m] for m in models_for_blend])

# Encontrar pesos ótimos não-negativos que somam 1 para minimizar o MSE
def loss_func(weights):
    w = weights / np.sum(weights)
    pred_blend = np.dot(blend_matrix, w)
    return mean_squared_error(y, pred_blend)

initial_weights = np.ones(len(models_for_blend)) / len(models_for_blend)
bounds = [(0, 1) for _ in range(len(models_for_blend))]
res = minimize(loss_func, initial_weights, bounds=bounds, method='SLSQP')
optimal_weights = res.x / np.sum(res.x)

oof_blend_preds = np.dot(blend_matrix, optimal_weights)
blend_val_mse = mean_squared_error(y, oof_blend_preds)
blend_val_rmse = np.sqrt(blend_val_mse)
blend_val_mae = mean_absolute_error(y, oof_blend_preds)
blend_val_r2 = r2_score(y, oof_blend_preds)
blend_val_mape = mean_absolute_percentage_error(y, oof_blend_preds) * 100

print(f"Pesos ótimos encontrados para o Blending:")
for m, w in zip(models_for_blend, optimal_weights):
    print(f"  - {m}: {w * 100:.2f}%")

cv_summary.append({
    "Modelo": "6. Blended Ensemble (RF + ExtraTrees + HistGB + GBR)",
    "MSE (Val)": blend_val_mse,
    "RMSE (Val)": blend_val_rmse,
    "MAE (Val)": blend_val_mae,
    "R² (Val)": blend_val_r2,
    "MAPE (%)": blend_val_mape,
    "MSE (Treino)": np.nan,
    "Razão Overfit": np.nan
})

# %% [markdown]
# # 8. Ranking Comparativo e Seleção do Pipeline Campeão

# %%
print("\n" + "=" * 85)
print("8. RANKING CONSOLIDADO DE MODELOS (ORDENADO PELO MENOR MSE)")
print("=" * 85)

df_rank = pd.DataFrame(cv_summary).sort_values(by="MSE (Val)", ascending=True).reset_index(drop=True)
df_rank.index = df_rank.index + 1
df_rank.index.name = "Rank"
print(df_rank.to_string())

best_model_name = df_rank.iloc[0]["Modelo"]
best_mse = df_rank.iloc[0]["MSE (Val)"]
best_rmse = df_rank.iloc[0]["RMSE (Val)"]
best_r2 = df_rank.iloc[0]["R² (Val)"]
baseline_mse = df_rank[df_rank["Modelo"].str.contains("Baseline RF")]["MSE (Val)"].values[0]
improvement_pct = ((baseline_mse - best_mse) / baseline_mse) * 100

print("\n" + "*" * 85)
print(f"  PIPELINE CAMPEÃO SELECIONADO: [{best_model_name}]")
print(f"  MSE de Validação Final:       {best_mse:.4f}")
print(f"  RMSE de Validação Final:      {best_rmse:.4f}")
print(f"  R² de Validação Final:        {best_r2:.4f}")
print(f"  Redução de Erro sobre Baseline: -{improvement_pct:.2f}% de MSE")
print("*" * 85)

# %% [markdown]
# # 9. Treinamento Final no Dataset Completo e Inferência no Teste

# %%
print("\n" + "=" * 85)
print("9. TREINAMENTO FINAL E INFERÊNCIA NO CONJUNTO DE TESTE (INTOCADO)")
print("=" * 85)

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Arquivo de teste não encontrado em {TEST_PATH}")

df_test = pd.read_csv(TEST_PATH)
print(f"[INFO] Conjunto de teste carregado de: {TEST_PATH}")
print(f"       Dimensões: {df_test.shape[0]:,} linhas x {df_test.shape[1]} colunas.")

X_test = df_test.drop(columns=[ID_COL]).copy()

# Estruturação da predição final baseada no campeão
if "Blended Ensemble" in best_model_name:
    print("\n[FIT] Treinando os modelos constituintes do Ensemble em 100% dos dados de treino...")
    test_preds_matrix = []
    for m_name in models_for_blend:
        config = candidate_models[m_name]
        pipeline = Pipeline([
            ('prep', build_preprocessor(apply_fe=config['apply_fe'])),
            ('reg', config['model'])
        ])
        pipeline.fit(X, y)
        test_preds_matrix.append(pipeline.predict(X_test))
    
    test_preds_matrix = np.column_stack(test_preds_matrix)
    final_test_predictions = np.dot(test_preds_matrix, optimal_weights)
else:
    print(f"\n[FIT] Treinando [{best_model_name}] em 100% dos dados de treino...")
    config = candidate_models[best_model_name]
    final_pipeline = Pipeline([
        ('prep', build_preprocessor(apply_fe=config['apply_fe'])),
        ('reg', config['model'])
    ])
    final_pipeline.fit(X, y)
    final_test_predictions = final_pipeline.predict(X_test)

# Garantir limite teórico do target [0, 100]
final_test_predictions = np.clip(final_test_predictions, 0.0, 100.0)

# Criar DataFrame de saída
df_pred_out = pd.DataFrame({
    ID_COL: df_test[ID_COL],
    f"{TARGET_COL}_Pred": final_test_predictions
})

print("\n" + "-" * 85)
print("ESTATÍSTICAS DESCRITIVAS DAS PREDIÇÕES GERADAS NO TESTE:")
print("-" * 85)
stats_pred = df_pred_out[f"{TARGET_COL}_Pred"].describe().to_frame(name="Estatísticas Predição Teste")
print(stats_pred.to_string())

# Salvar em ambos os arquivos para garantia de compatibilidade
df_pred_out.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
df_pred_out.to_csv(PREDICTIONS_OPT_PATH, index=False)
print(f"\n[SUCESSO] Predições salvas em: {PREDICTIONS_OUTPUT_PATH}")
print(f"[SUCESSO] Cópia otimizada salva em: {PREDICTIONS_OPT_PATH}")

print("\nPrimeiras 10 predições geradas no conjunto de teste:")
print(df_pred_out.head(10).to_string(index=False))

print("\n" + "=" * 85)
print("EXECUÇÃO DO PIPELINE AVANÇADO CONCLUÍDA COM SUCESSO!")
print("=" * 85)
