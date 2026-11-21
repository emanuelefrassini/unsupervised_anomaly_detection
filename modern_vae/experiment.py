"""Train on normal synthetic cases, select threshold on labeled calibration cases."""
import argparse
import copy
import json
import platform
from pathlib import Path
import numpy as np
import torch
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from .data import synthetic_cases,split_cases,Scaling,collate
from .model import MultimodalVAE,objective,scores


def calibrate(labels,values):
    labels=np.asarray(labels); values=np.asarray(values,dtype=float)
    if labels.ndim!=1 or labels.shape!=values.shape or set(labels)!={0,1} or not np.isfinite(values).all():
        raise ValueError('finite aligned calibration scores with both classes required')
    thresholds=np.r_[np.nextafter(values.min(),-np.inf),np.unique(values)]
    # Predict anomaly iff score > threshold; resolve ties with largest threshold.
    return float(max(thresholds,key=lambda t:(f1_score(labels,values>t,zero_division=0),t)))


def evaluate(labels,values,threshold):
    labels=np.asarray(labels); values=np.asarray(values); pred=values>threshold
    return dict(f1=float(f1_score(labels,pred,zero_division=0)),precision=float(precision_score(labels,pred,zero_division=0)),
                recall=float(recall_score(labels,pred,zero_division=0)),roc_auc=float(roc_auc_score(labels,values)),
                average_precision=float(average_precision_score(labels,values)))


def infer(model,cases):
    model.eval(); r,p,l,y=collate(cases)
    with torch.no_grad(): s=scores(model(r,p,l),r,p,l)
    return y.numpy(),s.numpy()


def run(n_normal=64,n_anomaly=32,epochs=12,seed=42,rgb_dim=12,pose_dim=6,hidden=32,latent=8,beta=.01):
    if epochs<1 or min(hidden,latent)<1 or beta<0: raise ValueError('invalid training configuration')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    cases=synthetic_cases(n_normal,n_anomaly,rgb_dim,pose_dim,seed)
    ids=split_cases(cases,seed+1)
    raw={k:[cases[i] for i in v] for k,v in ids.items()}
    scaler=Scaling.fit(raw['train']); data={k:[scaler.transform(c) for c in v] for k,v in raw.items()}
    model=MultimodalVAE(rgb_dim,pose_dim,hidden,latent)
    opt=torch.optim.Adam(model.parameters(),lr=.003); rng=np.random.default_rng(seed+2)
    best=float('inf'); state=None; best_epoch=0; history=[]
    for epoch in range(epochs):
        model.train(); order=rng.permutation(len(data['train'])); train_loss=0.
        for start in range(0,len(order),8):
            r,p,l,_=collate([data['train'][i] for i in order[start:start+8]])
            opt.zero_grad(); loss,_,_=objective(model(r,p,l),r,p,l,beta)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            train_loss+=float(loss.detach())*len(l)
        model.eval()
        with torch.no_grad():
            r,p,l,_=collate(data['validation']); v=float(objective(model(r,p,l),r,p,l,beta)[0])
        history.append({'epoch':epoch+1,'train_loss':train_loss/len(order),'validation_loss':v})
        if v<best: best=v; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
    if state is None: raise RuntimeError('no finite validation loss')
    model.load_state_dict(state)
    yc,sc=infer(model,data['calibration']); threshold=calibrate(yc,sc)
    yt,st=infer(model,data['test'])
    return {'scope':'synthetic only; not clinical validation or reproduction of published results','framework':torch.__version__, 'python':platform.python_version(), 'numpy':np.__version__, 'scikit_learn':sklearn.__version__,
            'config':dict(n_normal=n_normal,n_anomaly=n_anomaly,epochs=epochs,seed=seed,rgb_dim=rgb_dim,pose_dim=pose_dim,hidden=hidden,latent=latent,beta=beta),
            'split_ids':{k:v.tolist() for k,v in ids.items()}, 'scaling_fit_ids':ids['train'].tolist(),
            'best_epoch':best_epoch,'history':history,'threshold':threshold,
            'calibration':evaluate(yc,sc,threshold),'test':evaluate(yt,st,threshold),
            'held_out_scores':[{'synthetic_id':int(i),'label':int(y),'score':float(s)} for i,y,s in zip(ids['test'],yt,st)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal',type=int,default=64); p.add_argument('--anomalies',type=int,default=32)
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rgb-dim',type=int,default=12); p.add_argument('--pose-dim',type=int,default=6)
    p.add_argument('--hidden',type=int,default=32); p.add_argument('--latent',type=int,default=8); p.add_argument('--beta',type=float,default=.01)
    p.add_argument('--output',type=Path,default=Path('modern_vae_output.json'))
    a=p.parse_args(); r=run(a.normal,a.anomalies,a.epochs,a.seed,a.rgb_dim,a.pose_dim,a.hidden,a.latent,a.beta)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print('Synthetic held-out scores:',r['test']); print(f'Synthetic results only: {a.output}')
