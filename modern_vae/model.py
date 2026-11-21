"""Separate modality encoders and decoders with a shared Gaussian latent state."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class MultimodalVAE(nn.Module):
    def __init__(self,rgb_dim,pose_dim,hidden=32,latent=8):
        super().__init__()
        self.rgb_encoder=nn.LSTM(rgb_dim,hidden,batch_first=True)
        self.pose_encoder=nn.LSTM(pose_dim,hidden,batch_first=True)
        self.mean=nn.Linear(2*hidden,latent); self.logvar=nn.Linear(2*hidden,latent)
        self.rgb_decoder=nn.LSTM(latent,hidden,batch_first=True)
        self.pose_decoder=nn.LSTM(latent,hidden,batch_first=True)
        self.rgb_out=nn.Linear(hidden,rgb_dim); self.pose_out=nn.Linear(hidden,pose_dim)

    def encode(self,rgb,pose,lengths):
        def last(enc,x):
            packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_)=enc(packed)
            return h[-1]
        fused=torch.cat((last(self.rgb_encoder,rgb),last(self.pose_encoder,pose)),dim=-1)
        return self.mean(fused),self.logvar(fused).clamp(-12,12)

    def forward(self,rgb,pose,lengths,sample=None):
        mu,logvar=self.encode(rgb,pose,lengths)
        sample=self.training if sample is None else sample
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu) if sample else mu
        repeated=z[:,None,:].expand(-1,rgb.shape[1],-1)
        r,_=self.rgb_decoder(repeated); p,_=self.pose_decoder(repeated)
        return self.rgb_out(r),self.pose_out(p),mu,logvar


def per_case_mse(pred,target,lengths):
    valid=torch.arange(target.shape[1],device=target.device)[None,:]<lengths.to(target.device)[:,None]
    diff=(pred-target).masked_fill(~valid[:,:,None],0.)
    return diff.square().sum((1,2))/(lengths.to(target.device)*target.shape[-1])


def scores(output,rgb,pose,lengths):
    r,p,_,_=output
    return .5*(per_case_mse(r,rgb,lengths)+per_case_mse(p,pose,lengths))


def objective(output,rgb,pose,lengths,beta=.01):
    _,_,mu,logvar=output
    reconstruction=scores(output,rgb,pose,lengths).mean()
    kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(1).mean()
    return reconstruction+beta*kl,reconstruction,kl
