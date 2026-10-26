"""Treina e compara modelos para classificação binária da qualidade de vinhos.

Uso a partir da raiz do projeto:
    python src/train_model.py
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WineQT.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "quality" not in df.columns:
        raise ValueError("A base precisa conter a coluna 'quality'.")

    df["high_quality"] = (df["quality"] >= 7).astype(int)
    excluded = ["quality", "high_quality", "Id"]
    features = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in excluded
    ]
    X = df[features]
    y = df["high_quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Regressão Logística": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]),
    }

    rows, fitted, preds, probs = [], {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        fitted[name], preds[name], probs[name] = model, pred, prob
        rows.append({
            "Modelo": name,
            "Acurácia": accuracy_score(y_test, pred),
            "Precisão (alta qualidade)": precision_score(y_test, pred, zero_division=0),
            "Recall (alta qualidade)": recall_score(y_test, pred, zero_division=0),
            "F1-score (alta qualidade)": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })
        report = classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Baixa/média qualidade", "Alta qualidade"],
            zero_division=0, output_dict=True
        )
        pd.DataFrame(report).T.to_csv(
            RESULTS_DIR / f"relatorio_{'logistica' if name == 'Regressão Logística' else 'random_forest'}.csv"
        )

    metrics = pd.DataFrame(rows).set_index("Modelo").sort_values(
        "F1-score (alta qualidade)", ascending=False
    )
    metrics.to_csv(RESULTS_DIR / "comparacao_modelos.csv", encoding="utf-8-sig")
    print("\nComparação dos modelos:")
    print(metrics.round(3).to_string())

    # Matrizes de confusão
    fig, axes = plt.subplots(1, len(fitted), figsize=(11, 4))
    for ax, (name, pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Baixa/média", "Alta"],
            cmap="Blues", values_format="d", ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "matrizes_confusao.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Curva ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prob in probs.items():
        RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", label="Referência aleatória")
    ax.set_title("Curvas ROC — conjunto de teste")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curvas_roc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Importância das variáveis da Random Forest
    rf = fitted["Random Forest"].named_steps["model"]
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    importance.rename("importancia").to_csv(
        RESULTS_DIR / "importancia_variaveis.csv", encoding="utf-8-sig"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importance.head(10).values, y=importance.head(10).index, ax=ax)
    ax.set_title("Top 10 variáveis — importância da Random Forest")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "importancia_variaveis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "amostras": int(len(df)),
        "features": features,
        "proporcao_alta_qualidade": float(y.mean()),
        "melhor_modelo_por_f1": str(metrics.index[0]),
        "metricas": metrics.round(4).to_dict(orient="index"),
        "top_3_variaveis_random_forest": importance.head(3).index.tolist(),
    }
    (RESULTS_DIR / "resumo_execucao.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados salvos em: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
