import os
import yaml
from easydict import EasyDict

def create_config(config_file_exp):
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict(config)
   
    # Copy
    # for k, v in config.items():
    #     cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    # base_dir = os.path.join(root_dir, cfg['train_db_name'])
    # pretext_dir = os.path.join(base_dir, 'pretext')

    # cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    # cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    # cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    # cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    # if cfg['setup'] in ['scan', 'selflabel']:
    #     base_dir = os.path.join(root_dir, cfg['train_db_name'])
    #     scan_dir = os.path.join(base_dir, 'scan')
    #     selflabel_dir = os.path.join(base_dir, 'selflabel') 

        # cfg['scan_dir'] = scan_dir
        # cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        # cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        # cfg['selflabel_dir'] = selflabel_dir
        # cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        # cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')

    return cfg 

# Reference : https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/utils/config.py