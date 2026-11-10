from pathlib import Path
import numpy as np,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from common import ROOT
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,default=ROOT/"outputs/figures")
args=parser.parse_args()
W=args.output;W.mkdir(parents=True,exist_ok=True)
a=np.load(ROOT/'results/reference/probability_grid.npz');T=a['temperature'];t=a['time'];P=a['probability']
obs=list(csv.DictReader(open(ROOT/'data/training.csv',encoding='utf-8')))
points=list(csv.DictReader(open(ROOT/'results/reference/manuscript_test_predictions.csv',encoding='utf-8')))
plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6,'axes.linewidth':.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
def canvas(w,h):
 fig=plt.figure(figsize=(w/72,h/72));fig.patch.set_facecolor('white');return fig
def axes(fig,x,y,w,h,W,H):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
fig=canvas(230,142);ax=axes(fig,30,22,154.8,96.7,230,142)
cm=LinearSegmentedColormap.from_list('previous',np.array(json.loads((ROOT/'config/palette.json').read_text()))/255,N=256)
ax.contourf(T,t,P,levels=np.linspace(0,1,11),cmap=cm,vmin=0,vmax=1)
ax.contour(T,t,P,levels=[.5],colors=['#245e99'],linewidths=.6)
for cls,mk,col in [(0,'x','#8a4f2c'),(1,'o','#356684')]:
 rows=[r for r in obs if int(float(r['target']))==cls];ax.scatter([float(r['temperature']) for r in rows],[float(r['time']) for r in rows],s=8,marker=mk,c=col,linewidths=.6,zorder=4,clip_on=False)
ax.scatter([float(r['temperature_C']) for r in points],[float(r['time_h']) for r in points],marker='^',s=14,c='#ffe396',edgecolors='none',zorder=5)
off=[(1,-2),(1,-2),(1,2),(2,2),(-10,0),(-10,2)]
for r,ofs in zip(points,off):ax.annotate(r['point_id'],(float(r['temperature_C']),float(r['time_h'])),xytext=ofs,textcoords='offset points',fontsize=7,color='#375064',zorder=6)
ax.scatter([120],[10],marker='*',s=28,c='#f7bc50',edgecolors='none',zorder=6)
ax.set(xlim=(25,170),ylim=(0,24),xticks=[25,60,100,140,170],yticks=[0,6,12,18,24],xlabel='Temperature (°C)',ylabel='Time (h)');ax.tick_params(length=2,pad=2,width=.5);ax.xaxis.labelpad=2;ax.yaxis.labelpad=2
cbax=axes(fig,196,22,5.5,96.7,230,142);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cm),cax=cbax,ticks=[0,.25,.5,.75,1]);cb.ax.set_yticklabels(['0','0.25','0.5','0.75','1']);cb.ax.tick_params(labelsize=7,length=0,pad=2);cb.set_label('P(class 1)',fontsize=7,labelpad=2);cb.outline.set_visible(False)
h=[Patch(fc=cm(0)),Line2D([],[],marker='x',ls='',color='#8a4f2c'),Patch(fc=cm(1.0)),Line2D([],[],marker='o',ls='',color='#356684',markersize=3),Line2D([],[],color='#245e99',lw=.6),Line2D([],[],marker='^',ls='',color='#ffe396',markersize=4),Line2D([],[],marker='*',ls='',color='#f7bc50',markersize=5)]
fig.legend(h,['Class 0','Large Pd','Class 1','No large Pd','GPC, P = 0.5','Test points','Selected condition'],loc='upper left',bbox_to_anchor=(30/230,1),ncol=4,frameon=False,fontsize=6,columnspacing=.7,handlelength=1,handletextpad=.3,borderaxespad=0,labelspacing=.25)
for ext in ['pdf','png','svg']:fig.savefig(W/f'probability_map.{ext}',dpi=400)
plt.close(fig)
