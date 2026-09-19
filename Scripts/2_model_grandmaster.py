"""
===============================================================================
PROJETO: Impacto de IA nos Estudantes (AI Student Impact Dataset)
ARQUIVO: model_grandmaster.py
OBJETIVO: Pipeline Competitivo de Alta Performance (Grandmaster Pipeline)
          - Peer Group Deviations (desvios por curso e ano acadêmico)
          - Multi-Seed Bagging com 5 sementes nos modelos líderes de Boosting
          - Blindagem total contra Data Leakage em 5-Fold Cross Validation
          - Otimização de Pesos de Blending Out-of-Fold
          - Pós-processamento com calibração linear e tratamento do efeito teto (y=100)
          - Geração exclusiva do arquivo Database/test_predictions_grandmaster.csv
AUTOR: Senior Data Scientist Agent
DATA: 2026-09-17
===============================================================================
"""

# %% [markdown]
# # 1. Importação de Módulos e Configurações

# %%
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
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

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

TRAIN_PATH = "./Database/train.csv"
TEST_PATH = "./Database/test.csv"
PREDICTIONS_OUTPUT_PATH = "./Database/test_predictions_grandmaster.csv"
TARGET_COL = "Skill_Retention_Score"
ID_COL = "Student_ID"
BASE_RANDOM_STATE = 42
N_SPLITS = 5
SEEDS = [42, 123, 777, 999, 2026]

# %% [markdown]
# # 2. Engenharia de Features Avançada com Peer Deviations

# %%
ORDINAL_MAPPINGS = {
    'Year_of_Study': {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5},
    'Prompt_Engineering_Skill': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3},
    'Burnout_Risk_Level': {'Low': 1, 'Medium': 2, 'High': 3},
    'Institutional_Policy': {'Strict_Ban': 1, 'Allowed_With_Citation': 2, 'Actively_Encouraged': 3},
    'Paid_Subscription': {False: 0, True: 1}
}

class GrandmasterFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformador completo com:
    - Mapeamentos ordinais
    - Features de contexto de grupo (Peer Group Deviations) ajustadas no treino
    - Razões e interações de estudo e rendimento acadêmico
    """
    def __init__(self):
        self.peer_stats = {}
        
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy['Peer_Group'] = X_copy['Major_Category'].astype(str) + '_' + X_copy['Year_of_Study'].astype(str)
        
        # Calcular médias e desvios estritamente no fold de treino
        self.peer_stats['genai_mean'] = X_copy.groupby('Peer_Group')['Weekly_GenAI_Hours'].mean().to_dict()
        self.peer_stats['trad_mean'] = X_copy.groupby('Peer_Group')['Traditional_Study_Hours'].mean().to_dict()
        self.peer_stats['gpa_post_mean'] = X_copy.groupby('Peer_Group')['Post_Semester_GPA'].mean().to_dict()
        
        # Fallbacks globais
        self.global_genai_mean = X_copy['Weekly_GenAI_Hours'].mean()
        self.global_trad_mean = X_copy['Traditional_Study_Hours'].mean()
        self.global_gpa_mean = X_copy['Post_Semester_GPA'].mean()
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Encodings Ordinais
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in X_out.columns:
                X_out[col + '_Ord'] = X_out[col].map(mapping).fillna(0).astype(float)
                
        # 2. Interação de Alta Ordem Categórica
        X_out['Major_x_UseCase'] = X_out['Major_Category'].astype(str) + '__' + X_out['Primary_Use_Case'].astype(str)
        
        # 3. Peer Group Deviations (Mapeamento de estatísticas calculadas no fit)
        peer_group = X_out['Major_Category'].astype(str) + '_' + X_out['Year_of_Study'].astype(str)
        p_genai_mean = peer_group.map(self.peer_stats.get('genai_mean', {})).fillna(self.global_genai_mean)
        p_trad_mean = peer_group.map(self.peer_stats.get('trad_mean', {})).fillna(self.global_trad_mean)
        p_gpa_mean = peer_group.map(self.peer_stats.get('gpa_post_mean', {})).fillna(self.global_gpa_mean)
        
        X_out['GenAI_Diff_Peer'] = X_out['Weekly_GenAI_Hours'] - p_genai_mean
        X_out['Trad_Diff_Peer'] = X_out['Traditional_Study_Hours'] - p_trad_mean
        X_out['GPA_Ratio_Peer'] = X_out['Post_Semester_GPA'] / (p_gpa_mean + 1e-4)
        
        # 4. Relações de Estudo e Desempenho
        X_out['GPA_Delta'] = X_out['Post_Semester_GPA'] - X_out['Pre_Semester_GPA']
        X_out['Total_Study_Hours'] = X_out['Traditional_Study_Hours'] + X_out['Weekly_GenAI_Hours']
        X_out['Study_Ratio'] = X_out['Weekly_GenAI_Hours'] / (X_out['Traditional_Study_Hours'] + 1.0)
        X_out['Traditional_Study_Prop'] = X_out['Traditional_Study_Hours'] / (X_out['Total_Study_Hours'] + 1e-5)
        X_out['Study_Efficiency'] = X_out['GPA_Delta'] / (X_out['Total_Study_Hours'] + 1.0)
        
        # 5. Interações Comportamentais de IA e Estresse
        X_out['AI_Intensity_Index'] = X_out['Perceived_AI_Dependency'] * X_out['Weekly_GenAI_Hours']
        X_out['Exam_Stress_Impact'] = X_out['Anxiety_Level_During_Exams'] / (X_out['Pre_Semester_GPA'] + 0.1)
        X_out['Effective_AI_Usage'] = X_out['Prompt_Engineering_Skill_Ord'] * X_out['Weekly_GenAI_Hours']
        X_out['Tool_Efficiency'] = X_out['Tool_Diversity'] / (X_out['Weekly_GenAI_Hours'] + 1.0)
        X_out['Exam_Anxiety_per_Study_Hour'] = X_out['Anxiety_Level_During_Exams'] / (X_out['Traditional_Study_Hours'] + 1.0)
        
        return X_out

# %% [markdown]
# # 3. Definição do Pipeline de Pré-processamento

# %%
def build_grandmaster_preprocessor():
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
        ('fe', GrandmasterFeatureEngineering()),
        ('preprocessor', col_transformer)
    ])

# %% [markdown]
# # 4. Classe Multi-Seed Bagging Regressor

# %%
class MultiSeedRegressor(BaseEstimator):
    """
    Ensemble de sementes (Multi-Seed Bagging) para redução de variância estocástica.
    """
    def __init__(self, base_estimator_class, base_params, seeds):
        self.base_estimator_class = base_estimator_class
        self.base_params = base_params
        self.seeds = seeds
        self.models_ = []
        
    def fit(self, X, y):
        self.models_ = []
        for s in self.seeds:
            params = self.base_params.copy()
            params['random_state'] = s
            model = self.base_estimator_class(**params)
            model.fit(X, y)
            self.models_.append(model)
        return self
        
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for model in self.models_:
            preds += model.predict(X)
        return preds / len(self.models_)

# %% [markdown]
# # 5. Execução Principal: 5-Fold Cross Validation & Calibração

# %%
def main():
    print("=" * 85)
    print("  INICIANDO PIPELINE COMPETITIVO GRANDMASTER (MULTI-SEED & CALIBRAÇÃO)")
    print("=" * 85)
    
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_PATH}")
        
    df_train = pd.read_csv(TRAIN_PATH)
    X = df_train.drop(columns=[ID_COL, TARGET_COL]).copy()
    y = df_train[TARGET_COL].values
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=BASE_RANDOM_STATE)
    
    # 1. Definição dos Modelos Candidatos (com Multi-Seed Bagging nos campeões)
    models = {
        "Multi-Seed GBR (5 Seeds)": MultiSeedRegressor(
            GradientBoostingRegressor,
            {
                'n_estimators': 160,
                'learning_rate': 0.035,
                'max_depth': 5,
                'subsample': 0.85,
                'min_samples_leaf': 20
            },
            SEEDS
        ),
        "Multi-Seed HistGB (5 Seeds)": MultiSeedRegressor(
            HistGradientBoostingRegressor,
            {
                'max_iter': 160,
                'learning_rate': 0.035,
                'max_leaf_nodes': 31,
                'min_samples_leaf': 25,
                'l2_regularization': 3.0
            },
            SEEDS
        ),
        "ExtraTrees Regressor": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=15,
            max_features=0.7,
            random_state=BASE_RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    oof_predictions = {name: np.zeros(len(y)) for name in models.keys()}
    results_summary = []
    
    for name, model in models.items():
        print(f"\n--> Validando com 5-Fold CV: [{name}]...")
        fold_mses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            prep = build_grandmaster_preprocessor()
            X_tr_proc = prep.fit_transform(X_tr)
            X_va_proc = prep.transform(X_va)
            
            model.fit(X_tr_proc, y_tr)
            preds = model.predict(X_va_proc)
            
            oof_predictions[name][val_idx] = preds
            f_mse = mean_squared_error(y_va, preds)
            fold_mses.append(f_mse)
            
        mean_mse = np.mean(fold_mses)
        print(f"    MSE Médio OOF: {mean_mse:.4f} (± {np.std(fold_mses):.4f}) | RMSE: {np.sqrt(mean_mse):.4f}")
        results_summary.append({
            "Modelo": name,
            "MSE OOF": mean_mse,
            "RMSE OOF": np.sqrt(mean_mse),
            "MAE OOF": mean_absolute_error(y, oof_predictions[name]),
            "R² OOF": r2_score(y, oof_predictions[name])
        })
        
    # 2. Otimização do Blending Out-of-Fold
    print("\n" + "=" * 85)
    print("OTIMIZAÇÃO DOS PESOS DE BLENDING OUT-OF-FOLD")
    print("=" * 85)
    
    model_names = list(models.keys())
    blend_matrix = np.column_stack([oof_predictions[m] for m in model_names])
    
    def blend_loss(w):
        w_norm = w / np.sum(w)
        pred = np.dot(blend_matrix, w_norm)
        return mean_squared_error(y, pred)
        
    init_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    res_blend = minimize(blend_loss, init_w, bounds=bounds, method='SLSQP')
    opt_weights = res_blend.x / np.sum(res_blend.x)
    
    raw_blend_oof = np.dot(blend_matrix, opt_weights)
    raw_blend_mse = mean_squared_error(y, raw_blend_oof)
    print(f"MSE OOF Bruto do Blending: {raw_blend_mse:.4f}")
    for m, w in zip(model_names, opt_weights):
        print(f"  - {m}: {w * 100:.2f}%")
        
    # 3. Pós-Processamento e Calibração Linear + Efeito Teto
    print("\n" + "=" * 85)
    print("CALIBRAÇÃO DE SAÍDA E AJUSTE DO EFEITO TETO (y = 100.0)")
    print("=" * 85)
    
    def calib_func(params):
        a, b, stretch = params
        mod = raw_blend_oof * a + b
        mask = mod > 88.0
        mod[mask] = 88.0 + (mod[mask] - 88.0) * stretch
        mod = np.clip(mod, 0.0, 100.0)
        return mean_squared_error(y, mod)
        
    res_calib = minimize(calib_func, [1.01, -0.7, 1.05], bounds=[(0.95, 1.06), (-5.0, 5.0), (1.0, 1.15)], method='SLSQP')
    a_opt, b_opt, stretch_opt = res_calib.x
    
    calib_blend_oof = raw_blend_oof * a_opt + b_opt
    mask = calib_blend_oof > 88.0
    calib_blend_oof[mask] = 88.0 + (calib_blend_oof[mask] - 88.0) * stretch_opt
    calib_blend_oof = np.clip(calib_blend_oof, 0.0, 100.0)
    
    final_calib_mse = mean_squared_error(y, calib_blend_oof)
    final_calib_rmse = np.sqrt(final_calib_mse)
    final_calib_r2 = r2_score(y, calib_blend_oof)
    final_calib_mae = mean_absolute_error(y, calib_blend_oof)
    
    print(f"Parâmetros Ótimos de Calibração: a = {a_opt:.4f}, b = {b_opt:.4f}, stretch = {stretch_opt:.4f}")
    print(f"MSE Final Calibrado OOF:  {final_calib_mse:.4f} (Ganho de -{raw_blend_mse - final_calib_mse:.4f} no blending)")
    print(f"RMSE Final Calibrado OOF: {final_calib_rmse:.4f}")
    print(f"R² Final Calibrado OOF:   {final_calib_r2:.4f}")
    
    # 4. Treinamento Final em 100% dos Dados de Treino e Inferência no Teste
    print("\n" + "=" * 85)
    print("TREINAMENTO FINAL COMPLETO E INFERÊNCIA NO CONJUNTO DE TESTE")
    print("=" * 85)
    
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_PATH}")
        
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test.drop(columns=[ID_COL]).copy()
    
    final_prep = build_grandmaster_preprocessor()
    X_full_proc = final_prep.fit_transform(X)
    X_test_proc = final_prep.transform(X_test)
    
    test_preds_by_model = []
    for name in model_names:
        print(f"Ajustando [{name}] em 100% do dataset de treino...")
        m = models[name]
        m.fit(X_full_proc, y)
        test_preds_by_model.append(m.predict(X_test_proc))
        
    test_preds_matrix = np.column_stack(test_preds_by_model)
    raw_test_preds = np.dot(test_preds_matrix, opt_weights)
    
    # Aplicar calibração ótima
    calib_test_preds = raw_test_preds * a_opt + b_opt
    mask_test = calib_test_preds > 88.0
    calib_test_preds[mask_test] = 88.0 + (calib_test_preds[mask_test] - 88.0) * stretch_opt
    final_test_predictions = np.clip(calib_test_preds, 0.0, 100.0)
    
    # 5. Salvar EXCLUSIVAMENTE em Database/test_predictions_grandmaster.csv
    df_submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        f"{TARGET_COL}_Pred": final_test_predictions
    })
    
    df_submission.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"\n[SUCESSO] Novo arquivo de submissão salvo em: {PREDICTIONS_OUTPUT_PATH}")
    
    print("\n" + "-" * 85)
    print("ESTATÍSTICAS DESCRITIVAS DAS PREDIÇÕES DO NOVO FLUXO:")
    print("-" * 85)
    print(df_submission[f"{TARGET_COL}_Pred"].describe().to_frame(name="Grandmaster Preds").to_string())
    
    print("\nPrimeiras 10 predições geradas:")
    print(df_submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 85)
    print("PIPELINE GRANDMASTER CONCLUÍDO COM SUCESSO!")
    print("=" * 85)

if __name__ == '__main__':
    main()
