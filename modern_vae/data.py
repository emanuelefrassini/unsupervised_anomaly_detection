"""Paired synthetic sequences, disjoint case splits and training-only scaling."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class Case:
    rgb: np.ndarray
    pose: np.ndarray
    label: int


def synthetic_cases(n_normal=64,n_anomaly=32,rgb_dim=12,pose_dim=6,seed=42):
    if n_normal<10 or n_anomaly<4 or min(rgb_dim,pose_dim)<1:
        raise ValueError('insufficient cases or invalid dimensions')
    rng=np.random.default_rng(seed)
    wr=rng.normal(size=(3,rgb_dim)); wp=rng.normal(size=(3,pose_dim))
    cases=[]
    for i in range(n_normal+n_anomaly):
        length=int(rng.integers(16,33)); t=np.linspace(0,1,length)
        latent=np.column_stack((np.sin(2*np.pi*t),np.cos(2*np.pi*t),t))
        latent+=rng.normal(0,.12,size=latent.shape)
        label=int(i>=n_normal)
        r=latent@wr+rng.normal(0,.12,(length,rgb_dim))
        p=latent@wp+rng.normal(0,.12,(length,pose_dim))
        if label:
            # Deliberately artificial perturbation, not an actual catheter exchange.
            start=int(rng.integers(length//4,length//2)); end=min(length,start+length//3)
            p[start:end]+=rng.choice([-1.,1.],size=pose_dim)*3.5
            r[start:end]+=rng.choice([-1.,1.],size=rgb_dim)*2.5
        cases.append(Case(r.astype(np.float32),p.astype(np.float32),label))
    return cases


def split_cases(cases,seed=42):
    rng=np.random.default_rng(seed)
    labels=np.array([c.label for c in cases]); normal=rng.permutation(np.flatnonzero(labels==0)); abnormal=rng.permutation(np.flatnonzero(labels==1))
    if len(normal)<10 or len(abnormal)<4 or not np.isin(labels,[0,1]).all():
        raise ValueError('need at least ten normal and four abnormal cases')
    nt=int(.5*len(normal)); nv=max(1,int(.1*len(normal))); nc=(len(normal)-nt-nv)//2; ac=len(abnormal)//2
    return dict(train=normal[:nt], validation=normal[nt:nt+nv],
                calibration=np.concatenate((normal[nt+nv:nt+nv+nc],abnormal[:ac])),
                test=np.concatenate((normal[nt+nv+nc:],abnormal[ac:])))


@dataclass(frozen=True)
class Scaling:
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    pose_mean: np.ndarray
    pose_std: np.ndarray

    @classmethod
    def fit(cls,train):
        if not train or any(c.label!=0 for c in train): raise ValueError('fit only on normal training cases')
        r=np.concatenate([c.rgb for c in train]); p=np.concatenate([c.pose for c in train])
        return cls(r.mean(0),np.maximum(r.std(0),1e-6),p.mean(0),np.maximum(p.std(0),1e-6))

    def transform(self,c):
        return Case((c.rgb-self.rgb_mean)/self.rgb_std,(c.pose-self.pose_mean)/self.pose_std,c.label)


def collate(cases):
    lengths=torch.tensor([len(c.rgb) for c in cases],dtype=torch.long)
    if not len(cases) or any(len(c.rgb)!=len(c.pose) or len(c.rgb)<1 for c in cases):
        raise ValueError('paired nonempty aligned sequences required')
    n=len(cases); t=int(lengths.max())
    rgb=torch.zeros(n,t,cases[0].rgb.shape[1]); pose=torch.zeros(n,t,cases[0].pose.shape[1])
    for i,c in enumerate(cases):
        rgb[i,:lengths[i]]=torch.from_numpy(c.rgb); pose[i,:lengths[i]]=torch.from_numpy(c.pose)
    return rgb,pose,lengths,torch.tensor([c.label for c in cases],dtype=torch.long)
