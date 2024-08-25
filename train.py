import torch as t
from tqdm import tqdm
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from utils import entropy
from models import Transformer


device = 'cpu' #mps is way slower on Mac.


def compute_loss(model: Transformer, batch: t.Tensor, logits: t.Tensor, ent: t.Tensor, loss_fn, next_token: bool) -> t.Tensor:
    """
    Computes the loss of the model on a batch, and adds the entropy for normalization.
    If next_token=True, the predictions are compared with the next token.
    Otherwise, the predictions are compared with the full probability from pi.
    """

    batch = batch.to(device)
    model_proba = t.softmax(logits, dim=-1)
    pred_proba = model_proba[:, 1:-1, :]+1e-12

    if next_token:
        target = batch[:, 2:]
        loss = - ent + loss_fn(t.log(pred_proba.flatten(0, 1)), target.flatten(0, 1))
    else:
        true_proba = model.pi[2][batch[:, :-2], batch[:, 1:-1]].detach()
        loss = - ent - (true_proba*t.log(pred_proba)).sum(-1).mean()
    del batch
    return loss


def compute_acc(batch: t.Tensor, logits: t.Tensor) -> t.Tensor:
    """
    Compute the accuracy of the model for predicting the correct token.
    Meaningfull only if the task is to learn a look-up table, low-entropy distribution.
    """
    with t.no_grad():
        predictions = t.argmax(logits, dim=-1)
        target = batch[:, 2:]
        acc = (predictions[:, 1:-1] == target).to(t.float).mean()
    return acc


def train(model: Transformer, dataloader: DataLoader, lr: float=1e-3, next_token: bool=True) -> Dict[str, List[float]]:
    """
    Trains the model and return the loss and accuracy over all batches.
    If 'next_token', then the crossentropy is computed against the dirac of the next token,
    otherwise it is computed against the full distribution.
    """
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    ent = entropy(model.pi).to(device)
    loss_fn = t.nn.CrossEntropyLoss()

    Loss = []
    Acc = []
    for batch in tqdm(dataloader):
        logits = model(batch[0])[0]
        loss = compute_loss(model, batch[0], logits, ent, loss_fn, next_token)
        acc = compute_acc(batch[0], logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        Loss.append(loss.item())
        Acc.append(acc.item())
    model.to('cpu')
    
    return {'Loss': Loss, 'Acc': Acc}


def train_boosting(model: Transformer, dataloaders: List[DataLoader], lr: float=1e-3, next_token: bool=True) -> Dict[str, List[float]]:
    """
    Trains the model head by head by calling train.train.
    """
    para = model.meta_params['para']
    assert para > 1, "You only have one head, use train.train instead."  

    # Start by freezing all attention heads, they will be unfreezed and then re-freezed during the for loop
    freezer = {
        'freeze_E': False,
        'freeze_pos': False,
        'freeze_U': False,
        'freeze_Attn': [[{'freeze_O': True, 'freeze_Q': True, 'freeze_K': True, 'freeze_V': True} for _ in range(para)]]
    }
    model.freeze(freezer)

    # At initialisation, do not use any head, they will be added at each for loop
    model.skips['skip_attn'] = [[True for _ in range(para)]]

    attn_freezer: List[Dict[str, bool]] = freezer['freeze_Attn'][0]
    dict: Dict[str, List[float]] = {'Loss': [], 'Acc': []}
    for para in range(para):
        # Unskip head {para}
        model.skips['skip_attn'][0][para] = False

        # Unfreeze head {para}
        attn_freezer[para]['freeze_O'] = False
        attn_freezer[para]['freeze_QKV'] = False

        # Freeze all head before {para}, which have been trained already
        for ind in range(para):
            attn_freezer[ind]['freeze_O'] = True
            attn_freezer[ind]['freeze_QKV'] = True
        model.freeze(freezer)

        # Trains one head
        dict_para = train(model, dataloaders[para], lr, next_token=next_token)
        dict['Loss'] += dict_para['Loss']
        dict['Acc'] += dict_para['Acc']

        # Freezes the word embedding, positional embedding and unembedding once the first head is trained
        freezer['freeze_E'] = True
        freezer['freeze_U'] = True
        freezer['freeze_pos'] = True
    
    return dict