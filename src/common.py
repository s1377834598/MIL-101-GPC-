from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
COLS=['temperature','time']
KINDS=['RBF','Matern1.5','Matern2.5']
def build(p):
    scales=[p['temperature_scale'],p['time_scale']]
    k=RBF(scales,'fixed') if p['kernel']=='RBF' else Matern(scales,'fixed',nu=1.5 if p['kernel']=='Matern1.5' else 2.5)
    return make_pipeline(StandardScaler(),GaussianProcessClassifier(
        kernel=ConstantKernel(p['variance'],'fixed')*k,optimizer=None,random_state=42,max_iter_predict=200))
def load_data():
    d=pd.read_csv(ROOT/'data/training.csv')
    assert d.row_id.tolist()==list(range(len(d)))
    return d,json.loads((ROOT/'config/runs.json').read_text(encoding='utf-8'))
def load_models(folder=None):
    folder=Path(folder) if folder else ROOT/'models/frozen'
    files=[folder/f'MCCV_{i:02d}.joblib' for i in range(1,51)]
    if not all(p.is_file() for p in files):
        raise ValueError('All 50 models are required for the manuscript ensemble.')
    return [joblib.load(p) for p in files]
def predict(models,X,batch_size=2048):
    X=X[COLS]
    chunks=[]
    for start in range(0,len(X),batch_size):
        chunks.append(np.mean([m.predict_proba(X.iloc[start:start+batch_size])[:,1] for m in models],axis=0))
    return np.concatenate(chunks) if chunks else np.empty(0)
def metrics(y,p):
    return dict(accuracy=float(accuracy_score(y,p>=.5)),macro_F1=float(f1_score(y,p>=.5,average='macro')),
        Brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        ROC_AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else None)
