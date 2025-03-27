# iDeepG: Toward generalizable prediction of protein binding sites on RNAs using cross-attention networks with language models

## 1.Dependency
python=3.9.12  
torch==2.3.0+cu118

## 2.Predict
The pre-trained model weights for iDeepG can be downloaded from: http://www.csbio.sjtu.edu.cn/data/RBPsuite/val_model_epoch_12.pth

## 3. LucaOne inference checkpoint
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>

**Notice:**    
The project will download automatically LucaOne Trained-CheckPoint from **FTP**.
When downloading automatically failed, you can manually download:
Copy the **TrainedCheckPoint Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/TrainedCheckPoint/* </href> into the directory: `./models/llm/`

## Reference
Contact: Xiaoyong Pan (xypan172436@gmail.com)
