# iDeepG: Toward generalizable prediction of protein binding sites on RNAs using cross-attention networks with language models

## Dependency
python=3.9.12  
torch==2.3.0+cu118

## Predict
### Model Weights
The pre-trained model weights for iDeepG can be downloaded from: http://www.csbio.sjtu.edu.cn/data/RBPsuite/val_model_epoch_12.pth

### Command-Line Arguments
| Argument          | Type    | Description                                                                 |
|-------------------|---------|-----------------------------------------------------------------------------|
| `--rna_seq_path`  | str     | **Required**. Path to the input FASTA file containing RNA sequences         |
| `--prot_seq_path` | str     | **Required**. Path to the input FASTA file containing protein sequence     |
| `--device`        | str     | Device to use for computation (`cpu` or `cuda`). Default: automatically selects CUDA if available |
| `--save_path`     | str     | **Required**. Directory path where the output predictions will be saved    |
| `--model_weights` | str     | Path to the pre-trained model weights file. Default: `./val_model_epoch_12.pth` |

### Running Prediction
```bash
python iDeepG_predict.py --rna_seq_path rna_sequences.fasta --prot_seq_path protein_sequence.fasta --device cuda --save_path ./results
```

## LucaOne inference checkpoint
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>

**Notice:**    
The project will download automatically LucaOne Trained-CheckPoint from **FTP**.
When downloading automatically failed, you can manually download:
Copy the **TrainedCheckPoint Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/TrainedCheckPoint/* </href> into the directory: `./models/llm/`

## Reference
Contact: Xiaoyong Pan (2008xypan@sjtu.edu.cn)
