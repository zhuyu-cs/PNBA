from .pnba import PNBA

def get_io_dims(data_loader):
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)

def get_dims_for_loader_dict(dataloaders):
    return {k: get_io_dims(v) for k, v in dataloaders.items()}

def make_PNBA(
    dataloaders,
    proj_dict=dict(),
    encoder_decoder_dict=dict(),
    video_dict=dict()
):
    
    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    
    mices_dict = {k: v['responses'][-1] for k, v in session_shape_dict.items()}
    
    
    model = PNBA(
        mices_dict,
        proj_dict=proj_dict,
        encoder_decoder_dict=encoder_decoder_dict,
        video_dict=video_dict        
    )

    return model

