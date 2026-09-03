from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
code = lambda text: cells.append(nbf.v4.new_code_cell(text))

md("""# Desarrollo y productivización de un modelo inteligente para la detección de noticias falsas

**Big Data · Data Science · Business Intelligence · MLOps**

Este notebook implementa un flujo reproducible de extremo a extremo: entendimiento del problema, calidad de datos, modelado, evaluación probabilística, interpretabilidad, productivización, monitorización y gobierno.

> **Uso responsable:** el sistema estima patrones compatibles con las clases del conjunto de entrenamiento; no comprueba hechos ni sustituye una investigación periodística. La salida debe alimentar una revisión humana basada en fuente, autoría, fecha y evidencia.""")

md("""## 1. Objetivo de negocio y criterio de éxito

**Objetivo:** priorizar contenido que requiere verificación, reduciendo el esfuerzo manual y el tiempo de respuesta. La clase `0` es **falsa** y la clase `1` es **real**.

**KPIs técnicos:** F1 y recall de la clase falsa, ROC-AUC y Brier score (calidad probabilística).  
**KPIs operativos:** latencia, porcentaje de casos inciertos (40–70%), tasa de revisión humana, falsos negativos confirmados y deriva por idioma/fuente.  
**Decisión:** riesgo alto (≥70%) se envía a revisión prioritaria; 40–70% requiere revisión; <40% no significa que la noticia esté verificada.""")

md("""## 2. Preparación automática del entorno

Esta celda permite usar **Ejecutar todo** en Google Colab: instala únicamente las librerías ausentes. La base no tiene que subirse manualmente; la siguiente sección la descarga desde su fuente pública en Kaggle cuando `data/news.csv` no existe.""")
code("""import importlib.util, subprocess, sys

packages = {
    'pandas': 'pandas==2.3.3', 'numpy': 'numpy==2.5.2',
    'sklearn': 'scikit-learn==1.7.1', 'joblib': 'joblib==1.5.3',
    'matplotlib': 'matplotlib==3.10.5', 'seaborn': 'seaborn==0.13.2',
    'streamlit': 'streamlit==1.48.1', 'kagglehub': 'kagglehub>=0.4.3,<2'
}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
print('Entorno listo.')""")

code("""from pathlib import Path
import json, platform, sys, time, shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             classification_report, RocCurveDisplay)
from sklearn.calibration import CalibrationDisplay
from modeling import (build_pipeline, prepare_dataframe, save_bundle,
                      explain_prediction, fake_probability, get_feature_names)

SEED = 42
BASE = Path.cwd()
DATA_PATH = BASE / 'data' / 'news.csv'
MODEL_PATH = BASE / 'models' / 'fake_news_model.joblib'
FIGURES = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)
pd.set_option('display.max_colwidth', 120)
sns.set_theme(style='whitegrid', palette='deep')
print('Python:', sys.version.split()[0], '| Plataforma:', platform.system())
print('Dataset:', DATA_PATH)""")

md("## 3. Descarga automática, ingesta y auditoría inicial")
code("""if not DATA_PATH.exists():
    import kagglehub
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print('Descargando la base pública desde Kaggle…')
    try:
        downloaded = Path(kagglehub.dataset_download(
            'mucahiddemircan/real-and-fake-news-dataset',
            path='news.csv', output_dir=str(DATA_PATH.parent)
        ))
    except Exception:
        downloaded = Path(kagglehub.dataset_download(
            'mucahiddemircan/real-and-fake-news-dataset',
            output_dir=str(DATA_PATH.parent)
        ))
    candidates = ([downloaded] if downloaded.is_file() else []) + list(DATA_PATH.parent.rglob('news.csv'))
    if not candidates:
        raise FileNotFoundError('Kaggle descargó el recurso, pero no se encontró news.csv')
    source = candidates[0]
    if source.resolve() != DATA_PATH.resolve():
        shutil.copy2(source, DATA_PATH)
else:
    print('Se reutiliza la base local:', DATA_PATH)

raw = pd.read_csv(DATA_PATH)
print('Dimensiones:', raw.shape)
display(raw.head(3))
display(pd.DataFrame({
    'tipo': raw.dtypes.astype(str),
    'nulos': raw.isna().sum(),
    'únicos': raw.nunique(dropna=False)
}))""")

code("""label_counts = raw['label'].value_counts(dropna=False).sort_index()
ax = label_counts.rename(index={0: 'Falsa (0)', 1: 'Real (1)'}).plot.bar(color=['#d1495b', '#2a9d8f'])
ax.set(title='Distribución de clases', xlabel='', ylabel='Noticias')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=140)
plt.show()
print(label_counts)""")

md("""## 4. Limpieza y controles de calidad

Se aplica una limpieza **conservadora**: normalización Unicode, espacios, URL/correo y HTML. No se eliminan acentos, signos ni palabras funcionales porque contienen señales estilísticas útiles y permiten procesar español. También se eliminan nulos, textos demasiado cortos, etiquetas inválidas y duplicados antes de dividir los datos, evitando fuga de información.""")
code("""df, cleaning_report = prepare_dataframe(raw)
MAX_PER_CLASS = 4000
if df.groupby('label').size().max() > MAX_PER_CLASS:
    sampled = []
    for label, group in df.groupby('label'):
        sampled.append(group.sample(n=min(MAX_PER_CLASS, len(group)), random_state=SEED))
    df = pd.concat(sampled, ignore_index=True)
cleaning_report['rows_used_for_modeling'] = len(df)
display(pd.Series(cleaning_report, name='resultado'))
df['characters'] = df['text'].str.len()
display(df.groupby('label')['characters'].describe().round(1))
ax = sns.boxplot(data=df.sample(min(10000, len(df)), random_state=SEED), x='label', y='characters', showfliers=False)
ax.set(title='Longitud del texto por clase', xlabel='0 = falsa, 1 = real', ylabel='Caracteres')
plt.tight_layout(); plt.savefig(FIGURES / 'length_by_class.png', dpi=140); plt.show()""")

md("""## 5. Partición y prevención de fuga

La división es estratificada y reproducible (80/20). El vectorizador se ajusta **solo** con entrenamiento. Para una implantación real se recomienda además una prueba temporal o por fuente, ya que una división aleatoria puede sobreestimar la generalización cuando existen estilos editoriales persistentes.""")
code("""X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.20, stratify=df['label'], random_state=SEED
)
print('Entrenamiento:', X_train.shape[0], '| Prueba:', X_test.shape[0])
display(pd.crosstab(index=y_test, columns='test', normalize=True).round(4))""")

md("""## 6. Modelo bilingüe e interpretable

Se combinan TF-IDF de palabras (semántica local) y caracteres (morfología, estilo, robustez a variantes y transferencia parcial entre idiomas) con regresión logística. Es una arquitectura eficiente para Big Data disperso, ofrece probabilidades y permite explicar cada predicción mediante contribuciones de n-gramas.

**Alcance bilingüe honesto:** la interfaz admite inglés y español y los rasgos no dependen de un diccionario. Sin embargo, esta base es principalmente inglesa; el desempeño en español debe medirse con un corpus español etiquetado antes de uso crítico.""")
code("""model = build_pipeline()
if MODEL_PATH.exists():
    existing_bundle = joblib.load(MODEL_PATH)
    model = existing_bundle['model']
    training_seconds = existing_bundle['metadata']['metrics'].get('training_seconds', np.nan)
    print('Modelo productivo cargado:', MODEL_PATH)
else:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    print(f'Entrenamiento completado en {training_seconds:.1f} s')""")

md("## 7. Evaluación de clasificación y probabilidad")
code("""pred = model.predict(X_test)
fake_index = list(model.classes_).index(0)
p_fake = model.predict_proba(X_test)[:, fake_index]
y_fake = (y_test == 0).astype(int)

metrics = {
    'accuracy': accuracy_score(y_test, pred),
    'precision_fake': precision_score(y_test, pred, pos_label=0),
    'recall_fake': recall_score(y_test, pred, pos_label=0),
    'f1_fake': f1_score(y_test, pred, pos_label=0),
    'roc_auc_fake': roc_auc_score(y_fake, p_fake),
    'brier_fake': brier_score_loss(y_fake, p_fake),
    'test_rows': len(y_test),
    'training_seconds': round(training_seconds, 2),
}
display(pd.Series(metrics).to_frame('valor').style.format('{:.4f}'))
print(classification_report(y_test, pred, labels=[0, 1], target_names=['fake', 'real']))""")

code("""cm = confusion_matrix(y_test, pred, labels=[0, 1])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
            xticklabels=['Falsa', 'Real'], yticklabels=['Falsa', 'Real'])
axes[0].set(title='Matriz de confusión', xlabel='Predicción', ylabel='Real')
RocCurveDisplay.from_predictions(y_fake, p_fake, ax=axes[1], name='Modelo')
axes[1].plot([0,1], [0,1], '--', color='gray'); axes[1].set_title('ROC: clase falsa')
CalibrationDisplay.from_predictions(y_fake, p_fake, n_bins=10, strategy='quantile', ax=axes[2], name='Modelo')
axes[2].set_title('Calibración probabilística')
plt.tight_layout(); plt.savefig(FIGURES / 'model_evaluation.png', dpi=140); plt.show()""")

md("""## 8. Interpretabilidad global

Los coeficientes no demuestran causalidad: muestran asociaciones aprendidas. Términos de agencias o formatos editoriales pueden actuar como atajos, por lo que se deben auditar por fuente y periodo.""")
code("""feature_names = get_feature_names(model)
coef = model.named_steps['classifier'].coef_[0]
top_fake = np.argsort(coef)[:20]
top_real = np.argsort(coef)[-20:][::-1]
importance = pd.concat([
    pd.DataFrame({'feature': feature_names[top_fake], 'direction': 'fake', 'coefficient': coef[top_fake]}),
    pd.DataFrame({'feature': feature_names[top_real], 'direction': 'real', 'coefficient': coef[top_real]})
])
display(importance)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, idx, title, color in [(axes[0], top_fake, 'Señales hacia falsa', '#d1495b'), (axes[1], top_real, 'Señales hacia real', '#2a9d8f')]:
    vals = np.abs(coef[idx]); labels = [x.replace('word__','').replace('char__','') for x in feature_names[idx]]
    ax.barh(labels[::-1], vals[::-1], color=color); ax.set_title(title); ax.set_xlabel('|coeficiente|')
plt.tight_layout(); plt.savefig(FIGURES / 'global_interpretability.png', dpi=140); plt.show()""")

md("## 9. Interpretabilidad local y prueba bilingüe")
code("""examples = {
    'English': 'Scientists published a peer-reviewed study and provided the dataset and methodology for independent review.',
    'Español': 'Un mensaje viral asegura, sin citar fuentes ni aportar evidencia verificable, que un producto cura todas las enfermedades.'
}
rows = []
for language, article in examples.items():
    probability = fake_probability(model, article)
    rows.append({'language': language, 'probability_fake': probability, 'text': article})
    print('\n', language, f'P(falsa)={probability:.1%}')
    display(pd.DataFrame(explain_prediction(model, article, top_n=8)))
display(pd.DataFrame(rows))""")

md("""## 10. Análisis de errores y umbral de negocio

El umbral no debe elegirse solo por accuracy. En un caso donde dejar pasar una noticia falsa sea costoso, se baja el umbral para aumentar recall, aceptando más revisiones manuales.""")
code("""thresholds = np.arange(0.10, 0.91, 0.05)
threshold_table = []
for threshold in thresholds:
    predicted_fake = p_fake >= threshold
    threshold_table.append({
        'threshold': threshold,
        'precision_fake': precision_score(y_fake, predicted_fake, zero_division=0),
        'recall_fake': recall_score(y_fake, predicted_fake),
        'f1_fake': f1_score(y_fake, predicted_fake),
        'review_rate': predicted_fake.mean(),
    })
threshold_table = pd.DataFrame(threshold_table)
display(threshold_table.round(3))
threshold_table.plot(x='threshold', y=['precision_fake','recall_fake','f1_fake','review_rate'], figsize=(10,5))
plt.ylim(0,1.02); plt.title('Trade-off de umbral'); plt.tight_layout();
plt.savefig(FIGURES / 'threshold_tradeoff.png', dpi=140); plt.show()""")

code("""error_df = pd.DataFrame({'text': X_test, 'actual': y_test, 'prediction': pred, 'p_fake': p_fake})
errors = error_df[error_df['actual'] != error_df['prediction']].copy()
errors['error_type'] = np.where(errors['actual'].eq(0), 'falsa_no_detectada', 'real_marcada_falsa')
display(errors.groupby('error_type')['p_fake'].agg(['count','mean','min','max']).round(3))
display(errors.sort_values('p_fake').groupby('error_type').head(3)[['error_type','actual','prediction','p_fake','text']])""")

md("""## 11. Versionado y artefactos de producción

Se guarda un bundle con modelo y metadatos, métricas, informe de limpieza, manifiesto reproducible y hash SHA-256. El hash permite comprobar que el artefacto desplegado no cambió. La aplicación registra solo hash del texto, longitud, versión y score para reducir exposición de datos.""")
code("""metrics['confusion_matrix_labels_0_1'] = cm.tolist()
metadata = save_bundle(model, MODEL_PATH, metrics, cleaning_report)
report = classification_report(y_test, pred, labels=[0,1], target_names=['fake','real'], output_dict=True)
(MODEL_PATH.parent / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
manifest = {
    'python': sys.version, 'platform': platform.platform(),
    'sklearn': __import__('sklearn').__version__, 'pandas': pd.__version__,
    'seed': SEED, 'data_rows': len(df), 'model_sha256': metadata['sha256']
}
(MODEL_PATH.parent / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
display(pd.Series(metadata))
print('Modelo guardado en:', MODEL_PATH)""")

md("""## 12. Arquitectura de productivización y BI

```text
Fuentes → validación/limpieza → entrenamiento versionado → registro de modelo
                                                      ↓
Usuario → Streamlit/API → inferencia + explicación → revisión humana
                              ↓                 ↓
                      log sin texto      feedback verificado
                              ↓                 ↓
                         dashboard BI ← monitorización/deriva
```

**Dashboard BI recomendado:** volumen por día, distribución del score, casos por banda de riesgo, idioma, tasa de revisión, falsos positivos/negativos confirmados, latencia y versión.  
**Alertas:** cambio ≥10 puntos en casos inciertos, caída de F1/recall con etiquetas demoradas, PSI/KS de scores, exceso de errores o latencia.  
**Ciclo:** registrar → verificar una muestra → comparar por idioma/fuente → aprobar datos → reentrenar → validar → promover versión → rollback si falla.""")

md("""## 13. Interpretabilidad, sesgos, seguridad y gobierno

- **Interpretabilidad:** explicación global por coeficientes y local por contribuciones; usarla para detectar atajos como nombres de agencias.
- **Sesgo y deriva:** evaluar métricas separadas por idioma, fuente, tema y periodo. La ausencia de metadatos en esta base limita esa auditoría.
- **Privacidad:** no guardar el texto en logs; conservar solo hash y métricas técnicas salvo consentimiento y política explícita.
- **Seguridad:** limitar tamaño, sanitizar entradas, autenticar el servicio, cifrar tránsito y almacenamiento, y restringir acceso al modelo/logs.
- **Gobierno:** ficha de modelo, propietario, versión, aprobación humana, trazabilidad, caducidad, rollback y criterios de retirada.
- **Limitación central:** detección lingüística ≠ verificación factual. Para elevar la solución se debe integrar recuperación de fuentes confiables y contrastación de afirmaciones.

### Conclusión

El prototipo queda transformado en un activo reproducible y desplegable, con probabilidad, explicación, interfaz bilingüe, versionado y monitorización. El siguiente incremento de valor es construir y validar un corpus español representativo del contexto de la organización.""")

nb.cells = cells
path = BASE / 'Fake_News_Detection_End_to_End.ipynb'
nbf.write(nb, path)
print(path)
