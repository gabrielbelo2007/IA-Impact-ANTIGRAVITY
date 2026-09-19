"""
====================================================================================================
PROJETO: Impacto de IA na Retenção de Habilidades dos Estudantes (AI Student Impact)
ARQUIVO: pipeline_regression_template.py
OBJETIVO: Pipeline completo de Engenharia de Dados, EDA, Modelagem Comparativa de Regressão,
          Otimização de Hiperparâmetros, Ensembling com Blending OOF e Inferência de Teste.
          Transposição e aprimoramento integral do template 'Example/machine_learning_template.ipynb'
          com foco absoluto na minimização da métrica principal: MSE (Mean Squared Error).
AUTOR: Senior Data Scientist Agent
DATA: 2026-09-18
====================================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import randint, uniform

# Scikit-Learn
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
import joblib

warnings.filterwarnings("ignore")

# Configurações de exibição do Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# Constantes e caminhos
TRAIN_PATH = "./Database/train.csv"
TEST_PATH = "./Database/test.csv"
PREDICTIONS_DIR = "./Predictions"
PRED_OUTPUT_MAIN = os.path.join(PREDICTIONS_DIR, "test_predictions.csv")
PRED_OUTPUT_PIPELINE = os.path.join(PREDICTIONS_DIR, "test_predictions_template_pipeline.csv")
MODELS_DIR = "./models"
TARGET_COL = "Skill_Retention_Score"
ID_COL = "Student_ID"
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 90)
print("   PIPELINE COMPLETO DE REGRESSÃO & EDA (BASEADO NO TEMPLATE)")
print("   Métrica Principal de Avaliação e Otimização: MSE (Mean Squared Error)")
print("=" * 90)

# ==================================================================================================
# SEÇÃO 1: PRÉ-PROCESSAMENTO, HIGIENIZAÇÃO E ENGENHARIA DE ATRIBUTOS
# (Template Cells 18 a 93: Standardizing, Duplicates, Types, Missing, Outliers, Feature Engineering)
# ==================================================================================================

# Mapeamentos de categorias ordinais com lógica de progressão estrita
ORDINAL_MAPPINGS = {
    'Year_of_Study': {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5},
    'Prompt_Engineering_Skill': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3},
    'Burnout_Risk_Level': {'Low': 1, 'Medium': 2, 'High': 3},
    'Institutional_Policy': {'Strict_Ban': 1, 'Allowed_With_Citation': 2, 'Actively_Encouraged': 3},
    'Paid_Subscription': {False: 0, True: 1, 0: 0, 1: 1}
}

class TemplateFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformador customizado inspirado nas Cells 42-55 do template:
    - Mapeamento ordinal determinístico.
    - Extração de interações de domínio entre estudo e IA.
    - Peer Group Deviations calculados rigorosamente no .fit() de cada fold de treino
      para prevenir Data Leakage.
    """
    def __init__(self):
        self.peer_stats = {}
        self.global_genai_mean = 0.0
        self.global_trad_mean = 0.0
        self.global_gpa_mean = 0.0

    def fit(self, X, y=None):
        X_df = X.copy()
        peer_key = X_df['Major_Category'].astype(str) + '__' + X_df['Year_of_Study'].astype(str)
        X_df['Peer_Group'] = peer_key

        self.peer_stats['genai_mean'] = X_df.groupby('Peer_Group')['Weekly_GenAI_Hours'].mean().to_dict()
        self.peer_stats['trad_mean'] = X_df.groupby('Peer_Group')['Traditional_Study_Hours'].mean().to_dict()
        self.peer_stats['gpa_post_mean'] = X_df.groupby('Peer_Group')['Post_Semester_GPA'].mean().to_dict()

        self.global_genai_mean = float(X_df['Weekly_GenAI_Hours'].mean())
        self.global_trad_mean = float(X_df['Traditional_Study_Hours'].mean())
        self.global_gpa_mean = float(X_df['Post_Semester_GPA'].mean())
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Encodings Ordinais
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in X_out.columns:
                X_out[col + '_Ord'] = X_out[col].map(mapping).fillna(0).astype(float)

        # 2. Interação Categórica de Alta Ordem
        X_out['Major_x_UseCase'] = X_out['Major_Category'].astype(str) + '__' + X_out['Primary_Use_Case'].astype(str)

        # 3. Peer Group Deviations (desvios contextuais em relação à referência de treino)
        peer_group = X_out['Major_Category'].astype(str) + '__' + X_out['Year_of_Study'].astype(str)
        p_genai = peer_group.map(self.peer_stats.get('genai_mean', {})).fillna(self.global_genai_mean)
        p_trad = peer_group.map(self.peer_stats.get('trad_mean', {})).fillna(self.global_trad_mean)
        p_gpa = peer_group.map(self.peer_stats.get('gpa_post_mean', {})).fillna(self.global_gpa_mean)

        X_out['GenAI_Diff_Peer'] = X_out['Weekly_GenAI_Hours'] - p_genai
        X_out['Trad_Diff_Peer'] = X_out['Traditional_Study_Hours'] - p_trad
        X_out['GPA_Ratio_Peer'] = X_out['Post_Semester_GPA'] / (p_gpa + 1e-4)

        # 4. Relações de Estudo e Rendimento Acadêmico
        X_out['GPA_Delta'] = X_out['Post_Semester_GPA'] - X_out['Pre_Semester_GPA']
        X_out['Total_Study_Hours'] = X_out['Traditional_Study_Hours'] + X_out['Weekly_GenAI_Hours']
        X_out['Study_Ratio'] = X_out['Weekly_GenAI_Hours'] / (X_out['Traditional_Study_Hours'] + 1.0)
        X_out['Traditional_Study_Prop'] = X_out['Traditional_Study_Hours'] / (X_out['Total_Study_Hours'] + 1e-5)
        X_out['Study_Efficiency'] = X_out['GPA_Delta'] / (X_out['Total_Study_Hours'] + 1.0)

        # 5. Interações Comportamentais de IA e Fatores de Estresse
        X_out['AI_Intensity_Index'] = X_out['Perceived_AI_Dependency'] * X_out['Weekly_GenAI_Hours']
        X_out['Exam_Stress_Impact'] = X_out['Anxiety_Level_During_Exams'] / (X_out['Pre_Semester_GPA'] + 0.1)
        X_out['Effective_AI_Usage'] = X_out['Prompt_Engineering_Skill_Ord'] * X_out['Weekly_GenAI_Hours']
        X_out['Tool_Efficiency'] = X_out['Tool_Diversity'] / (X_out['Weekly_GenAI_Hours'] + 1.0)
        X_out['Exam_Anxiety_per_Study_Hour'] = X_out['Anxiety_Level_During_Exams'] / (X_out['Traditional_Study_Hours'] + 1.0)

        return X_out

def build_preprocessing_pipeline():
    """
    Monta o ColumnTransformer completo com RobustScaler para numéricas e OneHotEncoder para nominais.
    """
    nominal_cols = ['Major_Category', 'Primary_Use_Case', 'Major_x_UseCase']
    base_numeric = [
        'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity',
        'Traditional_Study_Hours', 'Perceived_AI_Dependency',
        'Anxiety_Level_During_Exams', 'Post_Semester_GPA'
    ]
    ord_numeric = [c + '_Ord' for c in ORDINAL_MAPPINGS.keys()]
    fe_numeric = [
        'GenAI_Diff_Peer', 'Trad_Diff_Peer', 'GPA_Ratio_Peer',
        'GPA_Delta', 'Total_Study_Hours', 'Study_Ratio',
        'Traditional_Study_Prop', 'Study_Efficiency',
        'AI_Intensity_Index', 'Exam_Stress_Impact',
        'Effective_AI_Usage', 'Tool_Efficiency', 'Exam_Anxiety_per_Study_Hour'
    ]
    all_numeric = base_numeric + ord_numeric + fe_numeric

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    col_trans = ColumnTransformer(
        transformers=[
            ('num', num_transformer, all_numeric),
            ('cat', cat_transformer, nominal_cols)
        ]
    )

    return Pipeline([
        ('fe', TemplateFeatureEngineering()),
        ('preprocessor', col_trans)
    ])


# ==================================================================================================
# SEÇÃO 2: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# (Template Cells 94 a 144: Univariate, Bivariate, Correlations, Cohen's d, Contingency)
# ==================================================================================================

def run_comprehensive_eda(df_train):
    print("\n" + "=" * 90)
    print("SEÇÃO 2: ANÁLISE EXPLORATÓRIA DE DADOS (EDA CONFORME TEMPLATE)")
    print("=" * 90)

    # 2.1 Univariada: Variável Alvo
    print(f"\n[EDA 2.1] Análise Univariada da Variável Alvo ({TARGET_COL}):")
    target_series = df_train[TARGET_COL]
    print(target_series.describe().to_frame().T)
    print(f"  Assimetria (Skewness): {target_series.skew():.4f}")
    print(f"  Curtose (Kurtosis):     {target_series.kurt():.4f}")
    print(f"  Efeito Teto (Valores == 100.0): {(target_series == 100.0).sum():,} amostras "
          f"({((target_series == 100.0).sum() / len(target_series)) * 100:.2f}%)")

    # 2.2 Univariada: Variáveis Numéricas
    num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in [ID_COL, TARGET_COL]]
    print(f"\n[EDA 2.2] Estatísticas Descritivas das Features Numéricas ({len(num_cols)} variáveis):")
    stats_num = df_train[num_cols].describe().T
    stats_num['skewness'] = df_train[num_cols].skew()
    stats_num['kurtosis'] = df_train[num_cols].kurt()
    print(stats_num[['mean', 'std', 'min', '50%', 'max', 'skewness', 'kurtosis']])

    # 2.3 Univariada: Variáveis Categóricas e Booleanas
    cat_cols = [c for c in df_train.columns if df_train[c].dtype in ['object', 'bool', 'string'] and c != TARGET_COL]
    print(f"\n[EDA 2.3] Distribuição de Frequências Categóricas ({len(cat_cols)} variáveis):")
    for col in cat_cols:
        counts = df_train[col].value_counts()
        props = df_train[col].value_counts(normalize=True) * 100
        dist = pd.DataFrame({'Total': counts, 'Percentual (%)': props})
        print(f"\n--- Atributo: [{col}] ---")
        print(dist.to_string())

    # 2.4 Bivariada: Correlações Numéricas com Target (Pearson e Spearman)
    print(f"\n[EDA 2.4] Correlações Lineares e Monotônicas com o Alvo ({TARGET_COL}):")
    pearson_corr = df_train[num_cols + [TARGET_COL]].corr(method='pearson')[TARGET_COL].drop(TARGET_COL)
    spearman_corr = df_train[num_cols + [TARGET_COL]].corr(method='spearman')[TARGET_COL].drop(TARGET_COL)
    corr_df = pd.DataFrame({
        'Pearson (Linear)': pearson_corr,
        'Spearman (Rank)': spearman_corr,
        'Pearson Abs': pearson_corr.abs()
    }).sort_values(by='Pearson Abs', ascending=False)
    print(corr_df[['Pearson (Linear)', 'Spearman (Rank)']])

    # 2.5 Bivariada: Médias e Efeito de Cohen's d por Categoria
    print("\n[EDA 2.5] Relação Categórica x Target e Tamanho de Efeito (Cohen's d):")
    for col in ['Prompt_Engineering_Skill', 'Burnout_Risk_Level', 'Major_Category']:
        grp = df_train.groupby(col)[TARGET_COL].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
        print(f"\nDesempenho por [{col}]:")
        print(grp)
        # Exemplo de Cohen's d entre primeiro e último grupo
        if len(grp) >= 2:
            g1_name, g2_name = grp.index[0], grp.index[-1]
            g1_data = df_train[df_train[col] == g1_name][TARGET_COL]
            g2_data = df_train[df_train[col] == g2_name][TARGET_COL]
            pooled_std = np.sqrt(((len(g1_data)-1)*g1_data.var() + (len(g2_data)-1)*g2_data.var()) / (len(g1_data)+len(g2_data)-2))
            cohens_d = (g1_data.mean() - g2_data.mean()) / (pooled_std + 1e-6)
            print(f"  -> Cohen's d [{g1_name} vs {g2_name}]: {cohens_d:.4f}")

    # 2.6 Auditoria de Outliers (Métodos 3SD, 1.5 IQR e Isolation Forest demonstrados no Template)
    print("\n[EDA 2.6] Auditoria de Detecção de Outliers nas Features Numéricas Chave:")
    for feat in ['Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Post_Semester_GPA']:
        mean_val, std_val = df_train[feat].mean(), df_train[feat].std()
        q1, q3 = df_train[feat].quantile(0.25), df_train[feat].quantile(0.75)
        iqr = q3 - q1
        outliers_3sd = ((df_train[feat] < mean_val - 3 * std_val) | (df_train[feat] > mean_val + 3 * std_val)).sum()
        outliers_iqr = ((df_train[feat] < q1 - 1.5 * iqr) | (df_train[feat] > q3 + 1.5 * iqr)).sum()
        print(f"  {feat:25s} | 3SD Outliers: {outliers_3sd:5d} ({outliers_3sd/len(df_train)*100:.2f}%) | "
              f"1.5 IQR Outliers: {outliers_iqr:5d} ({outliers_iqr/len(df_train)*100:.2f}%)")


# ==================================================================================================
# SEÇÃO 3: TREINAMENTO E VALIDAÇÃO DOS MODELOS DE BASELINE (REGRESSÃO)
# (Template Cells 145 a 166: Linear, ElasticNet, KNN, SVR, Tree, RandomForest, MLP, Gradient Boosters)
# ==================================================================================================

def run_baseline_regression_models(X_raw, y, kf):
    """
    Executa a comparação exaustiva de todos os algoritmos de regressão previstos no template,
    avaliando MSE (métrica principal), RMSE, MAE, R² e a razão de overfitting (MSE_Val / MSE_Treino).
    """
    print("\n" + "=" * 90)
    print("SEÇÃO 3: TREINAMENTO DOS MODELOS DE BASELINE DE REGRESSÃO (5-FOLD CV)")
    print("Métrica de Otimização e Ranqueamento Principal: MSE")
    print("=" * 90)

    # Dicionário cobrindo integralmente as classes de modelos do template:
    models = {
        "1. Linear Regression (OLS)": LinearRegression(),
        "2. Elastic Net Regressor": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
        "3. Ridge Regression (L2)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "4. K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1),
        "5. Support Vector (LinearSVR)": LinearSVR(C=1.0, random_state=RANDOM_STATE, max_iter=2000),
        "6. Decision Tree Regressor": DecisionTreeRegressor(max_depth=7, min_samples_leaf=20, random_state=RANDOM_STATE),
        "7. Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
        "8. Extra Trees Regressor": ExtraTreesRegressor(n_estimators=120, max_depth=14, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
        "9. Multi-Layer Perceptron (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, random_state=RANDOM_STATE, early_stopping=True),
        "10. HistGradientBoosting (HistGBR)": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=RANDOM_STATE),
        "11. Gradient Boosting (GBR)": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=RANDOM_STATE)
    }

    baseline_results = []
    oof_predictions = {name: np.zeros(len(y)) for name in models.keys()}

    for name, model in models.items():
        start_t = time.time()
        fold_val_mse = []
        fold_trn_mse = []
        fold_val_rmse = []
        fold_val_mae = []
        fold_val_r2 = []
        fold_val_mape = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw, y), 1):
            X_tr, y_tr = X_raw.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X_raw.iloc[val_idx], y[val_idx]

            # Pipeline com pré-processamento ajustado exclusivamente no fold de treino
            pipe = Pipeline([
                ('preprocessor', build_preprocessing_pipeline()),
                ('regressor', model)
            ])

            pipe.fit(X_tr, y_tr)
            pred_val = pipe.predict(X_val)
            pred_trn = pipe.predict(X_tr)

            oof_predictions[name][val_idx] = pred_val

            # Cálculo de Métricas no Fold
            v_mse = mean_squared_error(y_val, pred_val)
            t_mse = mean_squared_error(y_tr, pred_trn)
            v_rmse = np.sqrt(v_mse)
            v_mae = mean_absolute_error(y_val, pred_val)
            v_r2 = r2_score(y_val, pred_val)
            v_mape = mean_absolute_percentage_error(y_val, pred_val) * 100

            fold_val_mse.append(v_mse)
            fold_trn_mse.append(t_mse)
            fold_val_rmse.append(v_rmse)
            fold_val_mae.append(v_mae)
            fold_val_r2.append(v_r2)
            fold_val_mape.append(v_mape)

        elapsed = time.time() - start_t
        mean_val_mse = np.mean(fold_val_mse)
        mean_trn_mse = np.mean(fold_trn_mse)
        overfit_ratio = mean_val_mse / (mean_trn_mse + 1e-6)

        baseline_results.append({
            'Modelo': name,
            'MSE (Val)': mean_val_mse,
            'RMSE (Val)': np.mean(fold_val_rmse),
            'MAE (Val)': np.mean(fold_val_mae),
            'R² (Val)': np.mean(fold_val_r2),
            'MAPE (%)': np.mean(fold_val_mape),
            'MSE (Treino)': mean_trn_mse,
            'Razão Overfit': overfit_ratio,
            'Tempo (s)': elapsed
        })
        print(f"[{name:35s}] -> MSE Val: {mean_val_mse:.4f} | RMSE: {np.mean(fold_val_rmse):.4f} | R²: {np.mean(fold_val_r2):.4f} ({elapsed:.1f}s)")

    df_results = pd.DataFrame(baseline_results).sort_values(by='MSE (Val)', ascending=True).reset_index(drop=True)
    df_results.index = df_results.index + 1
    df_results.index.name = 'Rank'

    print("\n" + "-" * 90)
    print("QUADRO GERAL DE DESEMPENHO DOS MODELOS DE BASELINE (ORDENADO PELO MENOR MSE)")
    print("-" * 90)
    print(df_results.to_string())

    return df_results, oof_predictions


# ==================================================================================================
# SEÇÃO 4: OTIMIZAÇÃO DE HIPERPARÂMETROS (HYPERPARAMETER TUNING)
# (Template Cells 167 a 182: Grid Search & Randomized Search nos modelos campeões)
# ==================================================================================================

def run_hyperparameter_tuning(X_raw, y, kf):
    """
    Executa busca direcionada de hiperparâmetros para os modelos campeões baseados em árvore/boosting
    com critério exato de neg_mean_squared_error.
    """
    print("\n" + "=" * 90)
    print("SEÇÃO 4: OTIMIZAÇÃO FINA DE HIPERPARÂMETROS (CRITÉRIO: MENOR MSE)")
    print("=" * 90)

    # Pré-processamento dos dados completos para otimização rápida
    prep = build_preprocessing_pipeline()
    X_proc = prep.fit_transform(X_raw)

    tuning_summary = []

    # 4.1 Otimização de HistGradientBoostingRegressor
    print("\n--> 4.1. Otimizando HistGradientBoostingRegressor via Grid Search...")
    hist_grid = {
        'max_iter': [150, 200],
        'learning_rate': [0.03, 0.05],
        'max_leaf_nodes': [31, 45],
        'min_samples_leaf': [20, 30],
        'l2_regularization': [1.0, 3.0]
    }
    hist_search = GridSearchCV(
        estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid=hist_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    hist_search.fit(X_proc, y)
    best_hist_params = hist_search.best_params_
    best_hist_mse = -hist_search.best_score_
    print(f"    Melhor MSE (CV 3-Fold): {best_hist_mse:.4f}")
    print(f"    Melhores Hiperparâmetros: {best_hist_params}")

    tuning_summary.append({
        'Modelo': 'HistGradientBoostingRegressor',
        'Melhor MSE (CV)': best_hist_mse,
        'Hiperparâmetros': str(best_hist_params)
    })

    # 4.2 Otimização de GradientBoostingRegressor
    print("\n--> 4.2. Otimizando GradientBoostingRegressor...")
    gbr_params = {
        'n_estimators': 160,
        'learning_rate': 0.035,
        'max_depth': 5,
        'subsample': 0.85,
        'min_samples_leaf': 20,
        'random_state': RANDOM_STATE
    }
    gbr_model = GradientBoostingRegressor(**gbr_params)
    gbr_search_mse = -np.mean([
        -mean_squared_error(y[v], gbr_model.fit(X_proc[t], y[t]).predict(X_proc[v]))
        for t, v in list(kf.split(X_proc))[:3]
    ])
    print(f"    MSE Avaliado: {gbr_search_mse:.4f}")
    print(f"    Hiperparâmetros: {gbr_params}")

    tuning_summary.append({
        'Modelo': 'GradientBoostingRegressor',
        'Melhor MSE (CV)': gbr_search_mse,
        'Hiperparâmetros': str(gbr_params)
    })

    # 4.3 Otimização de Random Forest
    print("\n--> 4.3. Otimizando Random Forest...")
    rf_params = {
        'n_estimators': 160,
        'max_depth': 14,
        'min_samples_leaf': 8,
        'max_features': 0.75,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    rf_model = RandomForestRegressor(**rf_params)
    rf_search_mse = -np.mean([
        -mean_squared_error(y[v], rf_model.fit(X_proc[t], y[t]).predict(X_proc[v]))
        for t, v in list(kf.split(X_proc))[:3]
    ])
    print(f"    MSE Avaliado: {rf_search_mse:.4f}")
    print(f"    Hiperparâmetros: {rf_params}")

    tuning_summary.append({
        'Modelo': 'RandomForestRegressor',
        'Melhor MSE (CV)': rf_search_mse,
        'Hiperparâmetros': str(rf_params)
    })

    print("\n" + "-" * 90)
    print("RESUMO DA OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("-" * 90)
    print(pd.DataFrame(tuning_summary).to_string(index=False))

    return best_hist_params, gbr_params, rf_params


# ==================================================================================================
# SEÇÃO 5: MODELO FINAL, ENSEMBLE BLENDING & CALIBRAÇÃO (MINIMIZANDO MSE)
# (Template Cells 183 a 187: Final Model, Training, Evaluation, Ceiling Calibration)
# ==================================================================================================

class MultiSeedModel(BaseEstimator, RegressorMixin):
    """Bagging multi-semente para reduzir a variância estocástica e melhorar o MSE final."""
    def __init__(self, model_class, base_params, seeds=[42, 123, 777, 999, 2026]):
        self.model_class = model_class
        self.base_params = base_params
        self.seeds = seeds
        self.fitted_models_ = []

    def fit(self, X, y):
        self.fitted_models_ = []
        for s in self.seeds:
            p = self.base_params.copy()
            if 'random_state' in p:
                p['random_state'] = s
            m = self.model_class(**p)
            m.fit(X, y)
            self.fitted_models_.append(m)
        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for m in self.fitted_models_:
            preds += m.predict(X)
        return preds / len(self.fitted_models_)

def train_champion_ensemble(X_raw, y, kf, best_hist_params, gbr_params, rf_params):
    print("\n" + "=" * 90)
    print("SEÇÃO 5: CONSTRUÇÃO DO ENSEMBLE CAMPEÃO (BLENDING OUT-OF-FOLD & CALIBRAÇÃO)")
    print("Objetivo Matemático: Minimizar o MSE de Validação Cruzada")
    print("=" * 90)

    # Modelos constituintes de alta performance
    tuned_models = {
        "MultiSeed_HistGBR": MultiSeedModel(HistGradientBoostingRegressor, best_hist_params),
        "MultiSeed_GBR": MultiSeedModel(GradientBoostingRegressor, gbr_params),
        "MultiSeed_RF": MultiSeedModel(RandomForestRegressor, rf_params),
        "ExtraTrees_Tuned": ExtraTreesRegressor(n_estimators=160, max_depth=14, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1),
        "Ridge_Tuned": Ridge(alpha=5.0, random_state=RANDOM_STATE)
    }

    oof_matrix = np.zeros((len(y), len(tuned_models)))
    model_keys = list(tuned_models.keys())

    print("\n--> Treinando modelos constituintes em 5-Fold CV para gerar previsões Out-Of-Fold (OOF)...")
    for j, (m_name, model_obj) in enumerate(tuned_models.items()):
        start_m = time.time()
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw, y)):
            X_tr, y_tr = X_raw.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X_raw.iloc[val_idx], y[val_idx]

            prep = build_preprocessing_pipeline()
            X_tr_proc = prep.fit_transform(X_tr)
            X_val_proc = prep.transform(X_val)

            model_fold = MultiSeedModel(model_obj.model_class, model_obj.base_params) if isinstance(model_obj, MultiSeedModel) else model_obj
            model_fold.fit(X_tr_proc, y_tr)
            oof_matrix[val_idx, j] = model_fold.predict(X_val_proc)

        m_mse = mean_squared_error(y, oof_matrix[:, j])
        print(f"    [{m_name:20s}] -> OOF MSE: {m_mse:.4f} | RMSE: {np.sqrt(m_mse):.4f} ({time.time() - start_m:.1f}s)")

    # 5.2 Otimização Numérica dos Pesos do Blending (SLSQP minimizando MSE)
    print("\n--> Resolvendo Pesos Ótimos de Blending via SLSQP (Função Perda = MSE)...")
    def blend_loss(weights):
        w = weights / np.sum(weights)
        pred = np.dot(oof_matrix, w)
        return mean_squared_error(y, pred)

    init_w = np.ones(len(model_keys)) / len(model_keys)
    bounds = [(0.0, 1.0) for _ in range(len(model_keys))]
    res_blend = minimize(blend_loss, init_w, bounds=bounds, method='SLSQP')
    opt_weights = res_blend.x / np.sum(res_blend.x)

    raw_blend_oof = np.dot(oof_matrix, opt_weights)
    raw_blend_mse = mean_squared_error(y, raw_blend_oof)
    print(f"\n[Resultado Blending Bruto] MSE OOF: {raw_blend_mse:.4f} | RMSE OOF: {np.sqrt(raw_blend_mse):.4f}")
    print("Pesos Atribuídos:")
    for m_name, w in zip(model_keys, opt_weights):
        print(f"  * {m_name:20s}: {w * 100:6.2f}%")

    # 5.3 Calibração de Saída e Ajuste de Efeito Teto (y = 100.0)
    print("\n--> Otimizando Calibração Linear e Correção do Efeito Teto (minimiza MSE)...")
    def calib_loss(params):
        a, b, stretch = params
        mod = raw_blend_oof * a + b
        mask = mod > 88.0
        mod[mask] = 88.0 + (mod[mask] - 88.0) * stretch
        mod = np.clip(mod, 0.0, 100.0)
        return mean_squared_error(y, mod)

    res_calib = minimize(
        calib_loss,
        [1.01, -0.7, 1.05],
        bounds=[(0.95, 1.06), (-5.0, 5.0), (1.0, 1.15)],
        method='SLSQP'
    )
    a_opt, b_opt, stretch_opt = res_calib.x

    calib_blend_oof = raw_blend_oof * a_opt + b_opt
    mask = calib_blend_oof > 88.0
    calib_blend_oof[mask] = 88.0 + (calib_blend_oof[mask] - 88.0) * stretch_opt
    calib_blend_oof = np.clip(calib_blend_oof, 0.0, 100.0)

    final_calib_mse = mean_squared_error(y, calib_blend_oof)
    final_calib_rmse = np.sqrt(final_calib_mse)
    final_calib_r2 = r2_score(y, calib_blend_oof)
    final_calib_mae = mean_absolute_error(y, calib_blend_oof)
    final_calib_mape = mean_absolute_percentage_error(y, calib_blend_oof) * 100

    print(f"Parâmetros Ótimos de Calibração: Slope(a)={a_opt:.4f}, Intercept(b)={b_opt:.4f}, Ceiling Stretch={stretch_opt:.4f}")
    print("*" * 90)
    print(f"  DESEMPENHO FINAL DO PIPELINE CAMPEÃO (OOF - VALIDAÇÃO CRUZADA):")
    print(f"  MSE FINAL:  {final_calib_mse:.4f}  (Ganho líquido de -{raw_blend_mse - final_calib_mse:.4f} com calibração)")
    print(f"  RMSE FINAL: {final_calib_rmse:.4f}")
    print(f"  MAE FINAL:  {final_calib_mae:.4f}")
    print(f"  R² FINAL:   {final_calib_r2:.4f}")
    print(f"  MAPE FINAL: {final_calib_mape:.2f}%")
    print("*" * 90)

    return tuned_models, opt_weights, (a_opt, b_opt, stretch_opt), calib_blend_oof


# ==================================================================================================
# SEÇÃO 6: ANÁLISE DETALHADA DE RESÍDUOS E ERROS
# (Template Cells 188 a 197: Residual Plots, Error Distributions, Error vs Features Correlations)
# ==================================================================================================

def run_residual_and_error_analysis(X_raw, y, y_pred_oof):
    print("\n" + "=" * 90)
    print("SEÇÃO 6: ANÁLISE DETALHADA DE RESÍDUOS E ERROS (CONFORME TEMPLATE)")
    print("=" * 90)

    # Construção do DataFrame de análise de erros
    df_error = X_raw.copy()
    df_error['Actual'] = y
    df_error['Predicted'] = y_pred_oof
    df_error['Residual'] = df_error['Predicted'] - df_error['Actual']
    df_error['Absolute_Error'] = np.abs(df_error['Residual'])
    df_error['Squared_Error'] = df_error['Residual'] ** 2
    df_error['Percentage_Error'] = (df_error['Absolute_Error'] / (df_error['Actual'] + 1e-5)) * 100

    # 6.1 Estatísticas de Erro
    print("\n[Erro 6.1] Estatísticas Descritivas dos Erros:")
    print(f"  MSE Médio:                    {df_error['Squared_Error'].mean():.4f}")
    print(f"  RMSE Médio:                   {np.sqrt(df_error['Squared_Error'].mean()):.4f}")
    print(f"  Erro Absoluto Médio (MAE):    {df_error['Absolute_Error'].mean():.4f}")
    print(f"  Erro Absoluto Mediano:        {df_error['Absolute_Error'].median():.4f}")
    print(f"  Erro Percentual Médio (MAPE): {df_error['Percentage_Error'].mean():.2f}%")
    print(f"  Erro Percentual Mediano:      {df_error['Percentage_Error'].median():.2f}%")

    # 6.2 Distribuição dos Resíduos
    print("\n[Erro 6.2] Distribuição Estatística dos Resíduos:")
    res_stats = df_error['Residual'].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(res_stats.to_frame(name="Resíduo (Pred - Real)").T)
    print(f"  Assimetria dos Resíduos: {df_error['Residual'].skew():.4f} (Ideal ~ 0)")
    print(f"  Curtose dos Resíduos:    {df_error['Residual'].kurt():.4f}")

    # 6.3 Correlações entre Features e Erro Absoluto (Cell 196 do Template)
    print("\n[Erro 6.3] Correlações das Features com o Erro Absoluto (onde o modelo erra mais):")
    num_cols = df_error.select_dtypes(include=[np.number]).columns.tolist()
    features_to_check = [c for c in num_cols if c not in ['Actual', 'Predicted', 'Residual', 'Absolute_Error', 'Squared_Error', 'Percentage_Error']]
    corr_error = df_error[features_to_check + ['Absolute_Error']].corr()['Absolute_Error'].drop('Absolute_Error').sort_values(ascending=False)
    print(corr_error.to_frame(name="Correlação com Erro Absoluto"))

    # 6.4 Top 5 Melhores e Piores Casos de Predição (Cells 206-207 do Template)
    print("\n[Erro 6.4] Exemplos de Melhores Predições (Menor Erro Absoluto):")
    best_cases = df_error.sort_values(by='Absolute_Error', ascending=True).head(5)
    print(best_cases[['Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Post_Semester_GPA', 'Actual', 'Predicted', 'Absolute_Error']])

    print("\n[Erro 6.5] Exemplos de Piores Predições (Maior Erro Absoluto):")
    worst_cases = df_error.sort_values(by='Absolute_Error', ascending=False).head(5)
    print(worst_cases[['Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Post_Semester_GPA', 'Actual', 'Predicted', 'Absolute_Error']])


# ==================================================================================================
# SEÇÃO 7: IMPORTÂNCIA DE ATRIBUTOS E PERSISTÊNCIA DE MODELOS
# (Template Cells 198 a 210: Feature Importances, Linear Coefficients, Saving Pipelines)
# ==================================================================================================

def run_feature_importance_and_persistence(X_raw, y, tuned_models, opt_weights, calib_params):
    print("\n" + "=" * 90)
    print("SEÇÃO 7: IMPORTÂNCIA DE ATRIBUTOS E PERSISTÊNCIA DOS ARTEFATOS")
    print("=" * 90)

    # Ajuste final do pré-processador e modelo no dataset 100% completo
    final_prep = build_preprocessing_pipeline()
    X_full_proc = final_prep.fit_transform(X_raw)

    fitted_champions = {}
    model_keys = list(tuned_models.keys())

    print("\n--> Ajustando modelos finais constituintes em 100% dos dados de treino...")
    for m_name, model_obj in tuned_models.items():
        m_fit = MultiSeedModel(model_obj.model_class, model_obj.base_params) if isinstance(model_obj, MultiSeedModel) else model_obj
        m_fit.fit(X_full_proc, y)
        fitted_champions[m_name] = m_fit

    # Extração de Importância de Features (usando Random Forest e GBR)
    print("\n--> 7.1. Ranking Consolidado de Importância de Atributos:")
    rf_constituent = fitted_champions.get('MultiSeed_RF').fitted_models_[0]
    importances = rf_constituent.feature_importances_

    # Nomes das colunas pós-transformação
    try:
        col_trans = final_prep.named_steps['preprocessor']
        num_features = col_trans.transformers_[0][2]
        cat_features = col_trans.transformers_[1][1].named_steps['onehot'].get_feature_names_out(col_trans.transformers_[1][2])
        all_feature_names = list(num_features) + list(cat_features)
    except Exception:
        all_feature_names = [f"Feature_{i}" for i in range(len(importances))]

    df_importance = pd.DataFrame({
        'Feature': all_feature_names[:len(importances)],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print(df_importance.head(15).to_string())

    # Persistência dos artefatos (Pipeline, Pesos e Calibração)
    pipeline_artifact = {
        'preprocessor': final_prep,
        'fitted_models': fitted_champions,
        'weights': opt_weights,
        'calib_params': calib_params,
        'feature_importance': df_importance
    }

    save_path = os.path.join(MODELS_DIR, "final_regression_pipeline.joblib")
    joblib.dump(pipeline_artifact, save_path)
    print(f"\n[SUCESSO] Pipeline Campeão e modelos persistidos com sucesso em: {save_path}")

    return final_prep, fitted_champions


# ==================================================================================================
# SEÇÃO 8: INFERÊNCIA ÚNICA NO CONJUNTO DE TESTE E GERAÇÃO DO ARQUIVO DE PREDIÇÃO
# (Requisito Estrito: Database/test.csv utilizado apenas uma vez e salvo em Predictions/)
# ==================================================================================================

def run_single_test_inference_and_save(final_prep, fitted_champions, opt_weights, calib_params):
    print("\n" + "=" * 90)
    print("SEÇÃO 8: INFERÊNCIA FINAL ÚNICA NO CONJUNTO DE TESTE (Database/test.csv)")
    print("Gravação estrita das predições na pasta: ./Predictions/")
    print("=" * 90)

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Arquivo de teste não localizado em {TEST_PATH}")

    df_test = pd.read_csv(TEST_PATH)
    print(f"[INFO] Conjunto de teste carregado de: {TEST_PATH}")
    print(f"       Dimensões: {df_test.shape[0]:,} linhas x {df_test.shape[1]} colunas.")

    X_test = df_test.drop(columns=[ID_COL]).copy()

    # Aplicação estrita da transformação aprendida
    X_test_proc = final_prep.transform(X_test)

    # Predição com cada modelo constituinte
    preds_list = []
    for m_name in fitted_champions.keys():
        preds_list.append(fitted_champions[m_name].predict(X_test_proc))

    preds_matrix = np.column_stack(preds_list)
    raw_test_preds = np.dot(preds_matrix, opt_weights)

    # Aplicação dos parâmetros ótimos de calibração obtidos no OOF
    a_opt, b_opt, stretch_opt = calib_params
    calib_test_preds = raw_test_preds * a_opt + b_opt
    mask = calib_test_preds > 88.0
    calib_test_preds[mask] = 88.0 + (calib_test_preds[mask] - 88.0) * stretch_opt

    # Garantir limites teóricos do score [0.0, 100.0]
    final_test_predictions = np.clip(calib_test_preds, 0.0, 100.0)

    # Montagem do DataFrame de submissão
    df_submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        f"{TARGET_COL}_Pred": final_test_predictions
    })

    # Gravação nas saídas de Predictions/
    df_submission.to_csv(PRED_OUTPUT_MAIN, index=False)
    df_submission.to_csv(PRED_OUTPUT_PIPELINE, index=False)

    print(f"\n[SUCESSO] Arquivo de predição principal gravado em: {PRED_OUTPUT_MAIN}")
    print(f"[SUCESSO] Cópia identificada gravada em:           {PRED_OUTPUT_PIPELINE}")

    print("\n" + "-" * 90)
    print("ESTATÍSTICAS DESCRITIVAS DAS PREVISÕES GERADAS NO TESTE:")
    print("-" * 90)
    print(df_submission[f"{TARGET_COL}_Pred"].describe().to_frame(name="Estatísticas Predições Teste").to_string())

    print("\nPrimeiras 10 predições geradas para o conjunto de teste:")
    print(df_submission.head(10).to_string(index=False))

    print("\n" + "=" * 90)
    print("EXECUÇÃO DO PIPELINE DE REGRESSÃO FINALIZADA COM SUCESSO ABSOLUTO!")
    print("=" * 90)


# ==================================================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL
# ==================================================================================================

def main():
    start_total = time.time()

    # Carga dos dados de treino
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Arquivo não localizado em {TRAIN_PATH}")

    df_train = pd.read_csv(TRAIN_PATH)
    print(f"[INFO] Dados de treino carregados de: {TRAIN_PATH}")
    print(f"       Dimensões: {df_train.shape[0]:,} linhas x {df_train.shape[1]} colunas.")

    # 1. Executar EDA Completa
    run_comprehensive_eda(df_train)

    # 2. Separação de X e y
    X_raw = df_train.drop(columns=[ID_COL, TARGET_COL]).copy()
    y = df_train[TARGET_COL].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Modelos de Baseline de Regressão (5-Fold CV)
    baseline_df, oof_baselines = run_baseline_regression_models(X_raw, y, kf)

    # 4. Otimização de Hiperparâmetros
    best_hist_p, gbr_p, rf_p = run_hyperparameter_tuning(X_raw, y, kf)

    # 5. Ensemble Campeão, Blending SLSQP e Calibração
    tuned_models, opt_weights, calib_params, oof_preds = train_champion_ensemble(
        X_raw, y, kf, best_hist_p, gbr_p, rf_p
    )

    # 6. Análise de Resíduos e Erros
    run_residual_and_error_analysis(X_raw, y, oof_preds)

    # 7. Importância de Features e Persistência de Modelos
    final_prep, fitted_champions = run_feature_importance_and_persistence(
        X_raw, y, tuned_models, opt_weights, calib_params
    )

    # 8. Inferência Única no Conjunto de Teste e Gravação em Predictions/
    run_single_test_inference_and_save(final_prep, fitted_champions, opt_weights, calib_params)

    print(f"\nTempo Total de Execução do Pipeline: {(time.time() - start_total) / 60:.2f} minutos.")

if __name__ == "__main__":
    main()
