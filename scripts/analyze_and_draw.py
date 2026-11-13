from pathlib import Path
import hashlib
import json
import os
import sys

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "4"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

threadpool_limits(limits=4)
ROOT = Path(os.environ.get('ML_PROJECT_ROOT', '.')).resolve()
RUN = ROOT / 'outputs/course_study/cf_ml_neural_fusion_20261113_102759'
OUT = ROOT / 'outputs/reporting/prompt_revision_20261113'
DATA, FIGURES, QA = OUT / 'source_data', OUT / 'figures', OUT / 'qa'
for folder in [DATA, FIGURES, QA]:
    folder.mkdir(parents=True, exist_ok=True)

targets = ['adverse_outcome', 'readmission_30d']
main = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP_BatchNorm_Dropout_Res']
neural_ablations = ['MLP_BatchNorm_Dropout_Res', 'MLP_BatchNorm_Res', 'MLP_Dropout_Res',
                    'MLP_BatchNorm_Dropout', 'NeuralTaskSpecificFusion']
tree_ablations = ['CatBoost', 'TaskSpecificFusion']
models = main + [m for m in neural_ablations + tree_ablations if m not in main]
columns = ['encounter_id', 'fold'] + targets + [m + '__' + t for m in models for t in targets]
oof = pd.read_parquet(RUN / 'predictions/development_oof.parquet', columns=columns)
assert len(oof) == 700000 and oof.encounter_id.is_unique
existing_cv = pd.read_csv(RUN / 'cv_fold_metrics.csv').set_index(['method', 'fold'])
rows = []
for method in models:
    for fold, part in oof.groupby('fold', sort=True):
        item = {'method': method, 'fold': int(fold), 'n': len(part), 'threshold': 0.5}
        for label, target in [('A', targets[0]), ('R', targets[1])]:
            truth = part[target].to_numpy()
            probability = part[method + '__' + target].to_numpy()
            tn, fp, fn, tp = confusion_matrix(truth, probability >= 0.5, labels=[0, 1]).ravel()
            item['auc_' + label] = roc_auc_score(truth, probability)
            item['sen_' + label] = tp / (tp + fn)
            item['spe_' + label] = tn / (tn + fp)
            assert np.isclose(item['auc_' + label], existing_cv.loc[(method, fold), 'auc_' + label], atol=1e-12, rtol=0)
        for metric in ['auc', 'sen', 'spe']:
            item['macro_' + metric] = (item[metric + '_A'] + item[metric + '_R']) / 2
        rows.append(item)
fold_data = pd.DataFrame(rows)
fold_data.to_csv(DATA / 'cv_fold_fixed_05.csv', index=False)
stats = fold_data.groupby('method')[['auc_A', 'auc_R', 'macro_auc', 'macro_sen', 'macro_spe']].agg(['mean', 'std'])
stats.columns = ['_'.join(pair) for pair in stats.columns]
stats = stats.reset_index()
stats.to_csv(DATA / 'cv_fixed_05_summary.csv', index=False)

def contrast(method, reference):
    current = fold_data[fold_data.method.eq(method)].set_index('fold')
    base = fold_data[fold_data.method.eq(reference)].set_index('fold')
    result = {'method': method, 'reference': reference}
    for metric in ['auc_A', 'auc_R', 'macro_auc', 'macro_sen', 'macro_spe']:
        differences = current[metric] - base[metric]
        result[metric + '_delta_mean'] = differences.mean()
        result[metric + '_delta_sd'] = differences.std(ddof=1)
    return result

contrasts = [contrast('TaskSpecificFusion', 'CatBoost')]
contrasts += [contrast(method, neural_ablations[0]) for method in neural_ablations[1:]]
pd.DataFrame(contrasts).to_csv(DATA / 'paired_ablation_fixed_05.csv', index=False)

config = json.loads((RUN / 'configuration.json').read_text())
summary = json.loads((RUN / 'summary.json').read_text())
metrics = pd.read_csv(RUN / 'evaluation/holdout_metrics_with_intervals.csv')
metrics = metrics.set_index('target').loc[targets].reset_index()
probs = pd.read_parquet(RUN / 'predictions/holdout_probabilities.parquet')
truth_path = Path(config['release']['processed_directory']) / 'supplemental_locked.parquet'
truth = pd.read_parquet(truth_path, columns=['encounter_id'] + targets).set_index('encounter_id').loc[probs.encounter_id]
assert len(truth) == len(probs) == 100000 and probs.encounter_id.is_unique
for target in targets:
    metric = metrics.set_index('target').loc[target]
    values = roc_auc_score(truth[target], probs[target])
    assert np.isclose(values, metric.auc, atol=1e-12, rtol=0)
    cm = confusion_matrix(truth[target], probs[target] >= metric.threshold, labels=[0, 1])
    assert cm.tolist() == [[metric.tn, metric.fp], [metric.fn, metric.tp]]
metrics.to_csv(DATA / 'test_outcomes.csv', index=False)
macro = {'auc': float(metrics.auc.mean()), 'auc_lower': summary['holdout_macro_auc_interval'][0],
         'auc_upper': summary['holdout_macro_auc_interval'][1], 'sen': float(metrics.sensitivity.mean()),
         'spe': float(metrics.specificity.mean())}
assert np.isclose(macro['auc'], summary['holdout_macro_auc'], atol=1e-12, rtol=0)
macro['cm'] = [[macro['spe'], 1 - macro['spe']], [1 - macro['sen'], macro['sen']]]
pooled = {name: int(metrics[name].sum()) for name in ['tp', 'fp', 'tn', 'fn']}
pooled['sen'] = pooled['tp'] / (pooled['tp'] + pooled['fn'])
pooled['spe'] = pooled['tn'] / (pooled['tn'] + pooled['fp'])
(DATA / 'test_macro_micro.json').write_text(json.dumps({'macro': macro, 'pooled': pooled}, ensure_ascii=False, indent=2), encoding='utf-8')

# The drawn curves are sampled from every fold's ROC; exact AUC values use all scores.
# A dense common FPR grid retains a bounded, measured numerical display approximation.
GRID = np.linspace(0, 1, 10001)
curve_rows, area_errors = [], {}
for method in main:
    curves, aucs = [], []
    for target in targets:
        target_curves, target_aucs = [], []
        for fold, part in oof.groupby('fold', sort=True):
            fpr, tpr, _ = roc_curve(part[target], part[method + '__' + target], drop_intermediate=False)
            target_curves.append(np.interp(GRID, fpr, tpr))
            target_aucs.append(roc_auc_score(part[target], part[method + '__' + target]))
        curves.append(np.mean(target_curves, axis=0))
        aucs.append(float(np.mean(target_aucs)))
    macro_curve = np.mean(curves, axis=0)
    exact_auc = float(np.mean(aucs))
    area_error = abs(float(np.trapezoid(macro_curve, GRID)) - exact_auc)
    area_errors[method] = area_error
    assert area_error < 0.00006, (method, area_error)
    curve_rows.append(pd.DataFrame({'method': method, 'fpr': GRID, 'tpr_A': curves[0], 'tpr_R': curves[1],
                                    'macro_tpr': macro_curve, 'auc_A': aucs[0], 'auc_R': aucs[1], 'macro_auc': exact_auc}))
curves_cv = pd.concat(curve_rows, ignore_index=True)
curves_cv.to_csv(DATA / 'cv_mean_fold_macro_roc_grid.csv', index=False, float_format='%.9f')
old_curve_file = DATA / 'cv_pooled_oof_macro_roc_grid.csv'
if old_curve_file.exists():
    old_curve_file.unlink()

test_curves, test_aucs = [], []
for target in targets:
    fpr, tpr, _ = roc_curve(truth[target], probs[target], drop_intermediate=False)
    test_curves.append(np.interp(GRID, fpr, tpr))
    test_aucs.append(roc_auc_score(truth[target], probs[target]))
test_macro = np.mean(test_curves, axis=0)
area_errors['selected_test_macro'] = abs(float(np.trapezoid(test_macro, GRID)) - macro['auc'])
assert area_errors['selected_test_macro'] < 0.00006
pd.DataFrame({'fpr': GRID, 'tpr_A': test_curves[0], 'tpr_R': test_curves[1], 'macro_tpr': test_macro}).to_csv(
    DATA / 'test_macro_roc_grid.csv', index=False, float_format='%.9f')

for name in ['times.ttf', 'timesbd.ttf']:
    font_path = ROOT / 'assets/fonts' / name
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'path.simplify': False,
    'axes.spines.top': False, 'axes.spines.right': False})
BLUE, GOLD, GRID_COLOR = '#235F82', '#A8662C', '#B2B2B2'
WIDTH = 180 / 25.4
manifest = []


def export(fig, stem, title, legend, claim, sources):
    fig.canvas.draw()
    for extension in ['pdf', 'svg']:
        fig.savefig(FIGURES / (stem + '.' + extension), facecolor='white')
    fig.savefig(FIGURES / (stem + '.png'), dpi=600, facecolor='white')
    manifest.append({'stem': stem, 'number': len(manifest) + 1, 'title_zh': title, 'legend_zh': legend,
        'claim': claim, 'source_data': sources, 'width_mm': 180, 'height_mm': fig.get_figheight() * 25.4,
        'font': 'Times New Roman', 'smallest_nominal_font_pt': 7, 'destination': 'coursework manuscript'})
    plt.close(fig)


# Figure 1. The labels keep the selection flow visible at manuscript size.
fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.06, top=0.95)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.set_axis_off()

def node(x, y, w, h, text, color='#EAF2F5'):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.018',
                 linewidth=0.8, edgecolor=BLUE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8.2, linespacing=1.15)

def arrow(start, end):
    ax.add_patch(patches.FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8, linewidth=0.8, color=BLUE))

for args in [(0.01,0.70,0.18,0.19,'Training set\n700,000'), (0.25,0.70,0.18,0.19,'Five-fold CV'),
             (0.49,0.70,0.19,0.19,'Six algorithms'), (0.75,0.70,0.23,0.19,'Tree + MLP'),
             (0.75,0.39,0.23,0.19,'Fusion ablation'), (0.49,0.39,0.19,0.19,'Final model'),
             (0.25,0.39,0.18,0.19,'OOF thresholds'), (0.01,0.07,0.28,0.19,'Internal test\n100,000'),
             (0.38,0.07,0.28,0.19,'Kaggle test\n200,000')]:
    node(*args)
for start,end in [((.19,.795),(.25,.795)),((.43,.795),(.49,.795)),((.68,.795),(.75,.795)),
                  ((.865,.70),(.865,.58)),((.75,.485),(.68,.485)),((.49,.485),(.43,.485)),
                  ((.34,.39),(.15,.26)),((.585,.39),(.52,.26))]:
    arrow(start,end)
export(fig, 'figure_01_workflow', '研究流程',
       '训练集用于共同五折比较、模块消融和OOF阈值估计，内部测试集用于固定模型评价，平台测试集用于生成提交概率。树模型与MLP分别评价任务专属融合。',
       'Common-fold selection precedes final fixed-threshold evaluation.', ['cv_fixed_05_summary.csv', 'test_outcomes.csv'])

labels = {'LogisticRegression': 'Logistic regression', 'RandomForest': 'Random forest', 'XGBoost': 'XGBoost',
          'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost', 'MLP_BatchNorm_Dropout_Res': 'MLP'}
fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.5))
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.93, wspace=0.36, hspace=0.42)
for index, (axis, method) in enumerate(zip(axes.flat, main)):
    frame = curves_cv[curves_cv.method.eq(method)]
    axis.plot(GRID, frame.macro_tpr, color=BLUE, lw=1.35)
    axis.plot([0, 1], [0, 1], color=GRID_COLOR, lw=0.7, ls='--')
    axis.text(0.97, 0.07, 'AUC = %.4f' % frame.macro_auc.iloc[0], ha='right', va='bottom', fontsize=7.2,
              transform=axis.transAxes)
    axis.set_title(chr(ord('a') + index) + '  ' + labels[method], loc='left', fontweight='bold', pad=4)
    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[0,.5,1], yticks=[0,.5,1])
    if index >= 3:
        axis.set_xlabel('False positive rate')
    if index % 3 == 0:
        axis.set_ylabel('True positive rate')
export(fig, 'figure_02_cv_macro_roc', '各算法的共同五折Macro ROC曲线',
       'a—f依次为Logistic Regression、Random Forest、XGBoost、LightGBM、CatBoost和完整MLP。对每折、每项结局的ROC曲线按相同假阳性率插值，平均五折与两项结局的真阳性率。图内AUC为五折Macro AUC均值。',
       'Compare six algorithm configurations on the same cross-validated encounter predictions.', ['cv_mean_fold_macro_roc_grid.csv'])

fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.65))
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.86, wspace=0.34)
for i, (axis, tpr, title, auc) in enumerate(zip(axes, [*test_curves, test_macro],
        ['a  Outcome A', 'b  Outcome R', 'c  Macro'], [*test_aucs, macro['auc']])):
    axis.plot(GRID, tpr, color=BLUE, lw=1.35)
    axis.plot([0,1],[0,1], color=GRID_COLOR, lw=0.7, ls='--')
    if i < 2:
        row = metrics.iloc[i]
        axis.plot(1 - row.specificity, row.sensitivity, marker='o', color=GOLD, ms=3.7, zorder=4)
    axis.text(.97,.07,'AUC = %.4f' % auc,ha='right',va='bottom',fontsize=7.2,transform=axis.transAxes)
    axis.set_title(title, loc='left', fontweight='bold', pad=5)
    axis.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='False positive rate')
    if i == 0:
        axis.set_ylabel('True positive rate')
export(fig, 'figure_03_test_roc', '最终模型在内部测试集的ROC曲线',
       'a为ICU入住或院内死亡，b为30天非计划急诊再就诊，c为两项结局等权平均的Macro曲线。金色圆点表示训练集OOF锁定阈值对应的工作点。Macro曲线按相同假阳性率平均真阳性率。',
       'The final selected model performs differently for the two outcomes.', ['test_macro_roc_grid.csv', 'test_outcomes.csv'])

fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.65))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.85, wspace=0.42)
for i, (axis, title) in enumerate(zip(axes, ['a  Outcome A', 'b  Outcome R', 'c  Macro'])):
    axis.set(xlim=(0,2), ylim=(0,2), aspect='equal')
    axis.invert_yaxis()
    if i < 2:
        row = metrics.iloc[i]
        counts = np.array([[int(row.tn), int(row.fp)], [int(row.fn), int(row.tp)]])
        fractions = counts / counts.sum(axis=1, keepdims=True)
    else:
        counts = None
        fractions = np.array(macro['cm'])
    for r in range(2):
        for c in range(2):
            fraction = float(fractions[r, c])
            fill = (1 - .55 * fraction, 1 - .28 * fraction, 1 - .12 * fraction)
            axis.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=fill, edgecolor='white', lw=1.0))
            content = (f'{counts[r,c]:,}\n{fraction:.1%}' if counts is not None else f'{fraction:.1%}')
            axis.text(c+.5,r+.5,content,ha='center',va='center',fontsize=8.2,
                      color='white' if fraction > .76 else '#17222A',linespacing=1.2)
    axis.set_xticks([.5,1.5], ['0','1'])
    axis.set_yticks([.5,1.5], ['0','1'])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, loc='left',fontweight='bold',pad=5)
    axis.set_xlabel('Predicted')
    if i == 0:
        axis.set_ylabel('Observed')
export(fig, 'figure_04_confusion', '最终模型的任务别及Macro混淆矩阵',
       '任务A为ICU入住或院内死亡，任务R为30天非计划急诊再就诊。a、b给出计数及真实类别内百分比。c将相同真实类别中的对应比例等权平均。0表示结局未发生，1表示发生。',
       'Show outcome-specific counts and equal-outcome mean rates at the locked thresholds.', ['test_outcomes.csv', 'test_macro_micro.json'])

(OUT / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'analysis_summary.json').write_text(json.dumps({'run_id': RUN.name, 'selected_algorithm': 'CatBoost',
    'cv_fixed_threshold': 0.5, 'cv_threshold_role': 'descriptive SEN/SPE only; model selection remains mean fold Macro AUC',
    'macro': macro, 'pooled': pooled, 'rendered_roc_grid_error': area_errors,
    'macro_definition': 'arithmetic mean over two outcomes', 'cv_roc_basis': 'equal mean over five folds and two outcomes',
    'source_oof_sha256': hashlib.sha256((RUN / 'predictions/development_oof.parquet').read_bytes()).hexdigest()},
    ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUT), 'main_models': stats[stats.method.isin(main)].to_dict(orient='records'),
                  'contrasts': contrasts, 'macro': macro, 'pooled': pooled, 'area_errors': area_errors,
                  'figures': [record['stem'] for record in manifest]}, ensure_ascii=False))
