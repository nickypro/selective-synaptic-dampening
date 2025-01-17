import numpy as np
import torch
import einops
from tqdm import tqdm
from taker.data_classes import ActivationCollector, ActivationSummary
from taker.scoring import score_indices_by
from taker import Model
from welford_torch import Welford
import time

def get_midlayer_activations(model, dataset, token_limit=None):
    """Gets the activations of the midlayer ('key' layer) of MLPs for
    each layer, as well as for the pre_out layer of attention for each layer.
    """
    m = model.taker_model
    ff_abs   = Welford(dtype=m.dtype, device=m.device).detach()
    attn_abs = Welford(dtype=m.dtype, device=m.device).detach()

    tokens_seen = 0

    for batch in tqdm(dataset):
        # Get all necessary activations
        with torch.no_grad():
            t0 = time.time()
            try:
                pixel_values = batch["img"]
                _label       = batch["fine_label"]
                _output      = model(pixel_values=pixel_values)
                text_activations, residual_stream = [0,0,0,0], [0]
            except ValueError:
                print(f"Could not process an input")
                continue

            attn_act = m.get_attn_pre_out_activations(text_activations=text_activations, reshape=True).detach()
            attn_act = einops.rearrange(attn_act, 'layer token head pos -> token layer head pos')

            ff_act = m.get_ff_key_activations(residual_stream=residual_stream).detach()
            ff_act = einops.rearrange( ff_act, 'layer token pos -> token layer pos')

            ff_abs.add_all(ff_act.abs())
            attn_abs.add_all(attn_act.abs())

            tokens_seen += len(ff_act)
            if token_limit and tokens_seen > token_limit:
                break

    return {
        "ff":     ff_abs.mean,
        "attn": attn_abs.mean,
    }

def get_top_frac( values_tensor, top_frac: float ):
    # Get the number of entries in the tensor, and the number of entries to get
    shape = values_tensor.shape
    n_entries = np.prod(shape)
    k = int( top_frac * n_entries )

    # Get the top k values
    topk_values = torch.topk( values_tensor.flatten(), k, dim=-1, largest=True, sorted=False )

    # Create a criteria tensor with value 1 for all values in topk_values
    criteria = torch.zeros( n_entries, dtype=torch.bool )
    criteria[ topk_values.indices ] = True
    criteria = criteria.reshape( shape )

    # Get the threshold value, the value above which all values are in topk_values
    threshold = float( topk_values.values.flatten().min() )

    return criteria, threshold

def run_selective_pruning(model, retain_dl, forget_dl, frac=0.03):
    print("selective pruning frac = ", frac)
    # collect activations
    retain_act = get_midlayer_activations(model, retain_dl)
    forget_act = get_midlayer_activations(model, forget_dl)

    # score the neurons
    ff_scores   = forget_act["ff"]   /( retain_act["ff"]   + 1e-5)
    attn_scores = forget_act["attn"] /( retain_act["attn"] + 1e-5)

    # select the highest scoring neurons
    ff_top_neurons, _   = get_top_frac(ff_scores, frac)
    attn_top_neurons, _ = get_top_frac(attn_scores, frac)

    # delete the selected neurons
    model.taker_model.delete_ff_keys(ff_top_neurons)
    model.taker_model.delete_attn_pre_out(attn_top_neurons)

    return model
