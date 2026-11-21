import numpy as np
import pytest
import torch
from modern_vae.data import Case,synthetic_cases,split_cases,Scaling,collate
from modern_vae.model import MultimodalVAE,objective,scores,per_case_mse
from modern_vae.experiment import calibrate,evaluate,run

torch.set_num_threads(1)


def test_disjoint_splits():
    c=synthetic_cases(30,12,seed=2); s=split_cases(c,3)
    sets=[set(v) for v in s.values()]
    assert len(set.union(*sets))==len(c)==sum(map(len,sets))
    for k in ('train','validation'): assert all(c[i].label==0 for i in s[k])
    for k in ('calibration','test'): assert {c[i].label for i in s[k]}=={0,1}


def test_train_only_scaling():
    c=synthetic_cases(20,8); ids=split_cases(c)
    tr=[c[i] for i in ids['train']]; scaler=Scaling.fit(tr)
    r=np.concatenate([x.rgb for x in tr]); assert np.allclose(scaler.rgb_mean,r.mean(0))
    x=scaler.transform(tr[0]); assert np.allclose(x.rgb*scaler.rgb_std+scaler.rgb_mean,tr[0].rgb,atol=1e-6)
    with pytest.raises(ValueError): Scaling.fit([c[-1]])


def test_padding_invariance_and_two_stream_gradients():
    c=synthetic_cases(10,4,rgb_dim=5,pose_dim=3); r,p,l,_=collate([c[0],c[1],c[2]])
    m=MultimodalVAE(5,3,8,4).eval(); o=m(r,p,l)
    rr=torch.cat((r,torch.full((len(l),3,5),999.)),1); pp=torch.cat((p,torch.full((len(l),3,3),999.)),1)
    u=m(rr,pp,l)
    torch.testing.assert_close(o[2],u[2]); torch.testing.assert_close(scores(o,r,p,l),scores(u,rr,pp,l))
    # Both inputs affect the latent state.
    assert not torch.allclose(m(r+1,p,l)[2],o[2]); assert not torch.allclose(m(r,p+1,l)[2],o[2])
    m.train(); loss,rec,kl=objective(m(r,p,l),r,p,l); assert loss>=0 and rec>=0 and kl>=0
    loss.backward(); assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters())


def test_per_case_mse():
    y=torch.zeros(2,3,2); p=torch.tensor([[[1.,1.],[2.,2.],[99.,99.]],[[3.,3.],[99.,99.],[99.,99.]]]); l=torch.tensor([2,1])
    torch.testing.assert_close(per_case_mse(p,y,l),torch.tensor([2.5,9.]))


def test_calibration_ties_and_holdout():
    t=calibrate([0,0,1,1],[.1,.2,.8,.9]); assert t==.2
    assert evaluate([0,1],[.3,.9],t)['f1']==pytest.approx(2/3)
    with pytest.raises(ValueError): calibrate([0,0],[1.,2.])


def test_end_to_end():
    kw=dict(n_normal=20,n_anomaly=8,epochs=2,rgb_dim=5,pose_dim=3,hidden=8,latent=4)
    a=run(**kw); b=run(**kw); assert a==b
    assert a['split_ids']['train']==a['scaling_fit_ids']
    assert 0<=a['test']['f1']<=1 and 0<=a['test']['roc_auc']<=1
    assert {r['synthetic_id'] for r in a['held_out_scores']}==set(a['split_ids']['test'])
