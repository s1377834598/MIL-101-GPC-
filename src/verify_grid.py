import json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from common import ROOT,predict,load_models
g=np.load(ROOT/'results/reference/probability_grid.npz')
T,H=np.meshgrid(g['temperature'],g['time'])
with threadpool_limits(limits=1):
    P=predict(load_models(),pd.DataFrame({'temperature':T.ravel(),'time':H.ravel()})).reshape(T.shape)
err=float(np.max(np.abs(P-g['probability'])));assert err<1e-10
report=dict(passed=True,dataset='6.xlsx',stage='before',grid_points=int(P.size),max_probability_error=err)
(ROOT/'validation/grid_match.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
