import argparse
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from common import ROOT,load_models,predict
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=ROOT/'data/manuscript_test.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/predictions.csv')
    ap.add_argument('--models',type=Path,default=ROOT/'models/frozen')
    a=ap.parse_args();d=pd.read_csv(a.input)
    with threadpool_limits(limits=1):d['probability_good']=predict(load_models(a.models),d)
    d['predicted_class']=(d.probability_good>=.5).astype(int)
    a.output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.output,index=False)
    print(d.to_string(index=False))
if __name__=='__main__':main()
