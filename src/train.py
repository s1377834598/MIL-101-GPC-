"""Refit saved hyperparameters or repeat the original nested log-loss search."""
import argparse,json
from pathlib import Path
import joblib,optuna,pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from common import ROOT,COLS,KINDS,build,load_data,metrics

def search(X,y,seed):
    cv=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))
    def objective(t):
        p=dict(kernel=t.suggest_categorical('kernel',KINDS),
            variance=t.suggest_float('variance',.1,100.,log=True),
            temperature_scale=t.suggest_float('temperature_scale',.05,10.,log=True),
            time_scale=t.suggest_float('time_scale',.05,10.,log=True))
        prob=cross_val_predict(build(p),X,y,cv=cv,method='predict_proba')[:,1]
        return log_loss(y,prob,labels=[0,1])
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=15))
    study.enqueue_trial(dict(kernel='RBF',variance=1.,temperature_scale=1.,time_scale=1.))
    study.optimize(objective,n_trials=60)
    return study

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['fixed','search'],default='fixed',help='fixed: refit selected parameters; search: full 60-trial search per run')
    ap.add_argument('--limit',type=int,default=50,help='1..50; partial runs are for checks only')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/retrained')
    a=ap.parse_args()
    if not 1<=a.limit<=50: ap.error('--limit must be 1..50')
    if a.output.resolve().is_relative_to((ROOT/'models').resolve()): ap.error('Cannot overwrite shipped models')
    a.output.mkdir(parents=True,exist_ok=True)
    d,runs=load_data(); records=[]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with threadpool_limits(limits=1):
        for cfg in runs[:a.limit]:
            out=a.output/f"MCCV_{cfg['run']:02d}.joblib"
            if out.exists(): raise FileExistsError(f'{out}: choose a new output directory')
            tr,va=d.iloc[cfg['train_indices']],d.iloc[cfg['validation_indices']]
            p=cfg['best_params']
            if a.mode=='search':
                study=search(tr[COLS],tr.target,cfg['inner_seed']);p=study.best_params
                study.trials_dataframe().to_csv(a.output/f"trials_{cfg['run']:02d}.csv",index=False)
            model=build(p).fit(tr[COLS],tr.target)
            joblib.dump(model,out,compress=3)
            records.append(dict(run=cfg['run'],mode=a.mode,best_params=p,**metrics(va.target,model.predict_proba(va[COLS])[:,1])))
            (a.output/'training_report.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(f"Completed {cfg['run']}/{a.limit} ({a.mode})",flush=True)
if __name__=='__main__':main()
