import warnings

import numpy as np
import torch
from neuralpredictors.measures.np_functions import corr

def get_correlations(
    model_output,
    responses,
    per_neuron=True,
):
    """
    Computes single-trial correlation between model prediction and true responses
    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    # correlations = {}
    with torch.no_grad():
        # model_output : B t N
        responses = responses.detach().cpu()[:, -model_output.shape[1]:, :]
        model_output = model_output.detach().cpu()
        assert (
                responses.shape[-1] == model_output.shape[-1]
            ), f"model prediction is too short ({model_output.shape[-1]} vs {responses.shape[-1]})"
        

    responses = np.concatenate(list(responses), axis=1).T
    model_output = np.concatenate(list(model_output), axis=1).T    
    correlations = corr(responses, model_output, axis=0)

    if np.any(np.isnan(correlations)):
        warnings.warn(
            "{}% NaNs , NaNs will be set to Zero.".format(
                np.isnan(correlations).mean() * 100
            )
        )
    correlations[np.isnan(correlations)] = 0

    correlations = (
            np.hstack(correlations)
            if per_neuron
            else np.mean(np.hstack(correlations))
        )
    return correlations
