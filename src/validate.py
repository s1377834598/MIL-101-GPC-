"""Audit splits, reconstruct all models, and check manuscript predictions."""
import json,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import ROOT,COLS,build,load_data,load_models,predict,metrics
def main():
    d,runs=load_data();models=load_models();errors=[];fold_errors=[]
    manifest=pd.read_csv(ROOT/'data/outer_splits.csv')
    inner=pd.read_csv(ROOT/'data/inner_folds.csv')
    test=pd.read_csv(ROOT/'data/manuscript_test.csv')
    assert not set(map(tuple,d[COLS].to_numpy())) & set(map(tuple,test[COLS].to_numpy()))
    g=np.load(ROOT/'results/reference/probability_grid.npz')
    rows=np.unique(np.linspace(0,len(g['time'])-1,13,dtype=int));cols=np.unique(np.linspace(0,len(g['temperature'])-1,13,dtype=int))
    T,H=np.meshgrid(g['temperature'][cols],g['time'][rows]);grid=pd.DataFrame(dict(temperature=T.ravel(),time=H.ravel()))
    probe=pd.concat([d[COLS],test[COLS],grid],ignore_index=True)
    with threadpool_limits(limits=1):
        for cfg,model in zip(runs,models):
            tr,va=cfg['train_indices'],cfg['validation_indices']
            assert len(tr)==18 and len(va)==6 and not set(tr)&set(va)
            assert sorted(tr+va)==list(range(len(d)))
            assert d.iloc[va].target.value_counts().to_dict()=={0:3,1:3}
            for role,ids in [('train',tr),('validation',va)]:
                f=manifest[(manifest.run==cfg['run'])&(manifest.role==role)]
                assert f.row_id.tolist()==ids
            for fold,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=cfg['inner_seed']).split(d.iloc[tr][COLS],d.iloc[tr].target),1):
                for role,ids in [('train',it),('validation',iv)]:
                    f=inner[(inner.run==cfg['run'])&(inner.fold==fold)&(inner.role==role)]
                    assert f.row_id.tolist()==[tr[int(j)] for j in ids]
            rebuilt=build(cfg['best_params']).fit(d.iloc[tr][COLS],d.iloc[tr].target)
            errors.append(float(np.max(np.abs(rebuilt.predict_proba(probe[COLS])[:,1]-model.predict_proba(probe[COLS])[:,1]))))
        p=predict(models,test);expected=pd.read_csv(ROOT/'results/reference/manuscript_test_predictions.csv').probability_class1_ensemble.to_numpy()
        np.testing.assert_allclose(p,expected,atol=1e-10,rtol=0)
        grid_error=float(np.max(np.abs(predict(models,grid)-g['probability'][np.ix_(rows,cols)].ravel())))
        assert grid_error<1e-10 and max(errors)<1e-10
        p120=float(predict(models,pd.DataFrame({'temperature':[120.],'time':[10.]}))[0])
    hashes=json.loads((ROOT/'provenance/package_sha256.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    report=dict(passed=True,models=50,unique_training_conditions=len(d),outer_splits=50,inner_folds=150,
        model_refit_max_probability_error=max(errors),grid_sample_points=len(grid),grid_max_probability_error=grid_error,
        six_point_max_probability_error=float(np.max(np.abs(p-expected))),six_point_metrics=metrics(test.target,p),
        probability_120C_10h=p120,package_checksums_verified=len(hashes),
        limitations=['Checks establish numerical reproducibility, not independent experimental validity.',
            'Full 50-run hyperparameter search was not rerun by this validation script.',
            'Six conditions match their labels; method selection independence is not established.'])
    (ROOT/'validation/report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
