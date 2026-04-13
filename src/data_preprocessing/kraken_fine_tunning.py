# segtrain.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))

import rich.console
_original_clear_live = rich.console.Console.clear_live
def _safe_clear_live(self):
    try:
        _original_clear_live(self)
    except IndexError:
        pass
rich.console.Console.clear_live = _safe_clear_live

import lightning.pytorch as pl
from kraken.lib.train import SegmentationModel

train_file = r"D:/Users/kju10/Documents/LMU-STATISTICS & DATA SCIENCE MASTER/SS2026/Thesis/OCC_HTR/data/processed/annotated_samples/05_garde_001.xml"
val_file   = r"D:/Users/kju10/Documents/LMU-STATISTICS & DATA SCIENCE MASTER/SS2026/Thesis/OCC_HTR/data/processed/annotated_samples/17_f_012v_013.xml"
model_path = r"D:/Users/kju10/Documents/LMU-STATISTICS & DATA SCIENCE MASTER/SS2026/Thesis/OCC_HTR/models/segmentation/blla.mlmodel"
output     = "models/segmentation/my_occitan_segmentation"

hyper_params = {
    'line_width': 8,
    'padding': (0, 0),
    'freq': 1.0,
    'quit': 'fixed',
    'epochs': 5,
    'min_epochs': 0,
    'lag': 10,
    'min_delta': None,
    'optimizer': 'Adam',
    'lrate': 2e-4,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'warmup': 0,
    'schedule': 'constant',
    'gamma': 0.1,
    'step_size': 1,
    'sched_patience': 5,
    'cos_max': 50,
    'cos_min_lr': 2e-5,
    'batch_size': 1,
    'augment': False,
}

if __name__ == '__main__':
    model = SegmentationModel(
        hyper_params,
        output=output,
        model=model_path,
        training_data=[train_file],
        evaluation_data=[val_file],
        format_type='xml',
        partition=0.9,
        num_workers=0,             
        force_binarization=False,
        suppress_regions=True,     
        suppress_baselines=False,
        resize="both", 
        valid_regions=None,
        valid_baselines=None,
        merge_regions=None,
        merge_baselines=None,
        bounding_regions=None,
        topline=False,
    )

    print(f"Training samples: {len(model.train_set)}")
    print(f"Validation samples: {len(model.val_set)}")

    # DEBUG: test dataset BEFORE training
    print("\nTesting dataset access...")
    sample = model.train_set[0]
    print("Sample loaded successfully!")

    trainer = pl.Trainer(
        max_epochs=hyper_params['epochs'],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[],
        accelerator='cpu',
        #devices=1,
        precision="16-mixed", 
        num_sanity_val_steps=0,  
    )

    print("Starting training...")
    trainer.fit(model)

    print(f"Best epoch: {model.best_epoch}")

    import glob
    saved = glob.glob(f"{output}*")
    print(f"Saved files: {saved}")