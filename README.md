## Disentangling Urban Flow: A Dynamic ST-GNN Approach Based on Multi-View Contrastive Learning

This repository implements the D2STCNet model for spatiotemporal traffic prediction using deep learning techniques. 

The paper was accepted by CSCWD(IEEE International Conference on Computer Supported Cooperative Work in Design)2025. 

### Project structure
```bash
D2STCNet/
│
├── configs/
│   └── .yaml              # Configuration file for NYCBike1 dataset
│
├── data/
│   └── DataSets
│
├── lib/
│   ├── dataloader.py             # Data loading functions
│   ├── logger.py                 # Logging utility
│   ├── metrics.py                # Performance metrics computation
│   └── utils.py                  # Utility functions
│
├── model/
│   ├── aug.py                    # Data augmentation techniques
│   ├── layers.py                 # Custom neural network layers
│   ├── models.py                 # Model definitions
│   └── trainer.py                # Training logic and loops
│
├── .gitignore                    # Git ignore file
├── main.py                       # Main script to train and evaluate the model
└── README.md                     # Documentation for the project
```

### Requirements
* Python 3.10 or later
* PyTorch 1.10.0 or later
* NumPy 1.20.0 or later

> extra: If you want to perform automated hyperparameter tuning, please install Optuna.

### Run Code
```bash
cd /path/to/target/folder
export CUDA_VISIBLE_DEVICES=$1
source activate yourvenv
python main.py --config_filename=configs/xx.yaml
```
> Our experimental platform uses A100 GPU

### Processed Data or Raw Data
You can download the preprocessed dataset from our [Google Driver](https://drive.google.com/drive/folders/1giE8LOXsmobcIoknV9AyigVhezEdisdW?dmr=1&ec=wgc-drive-globalnav-goto&q=sharedwith:public%20parent:1giE8LOXsmobcIoknV9AyigVhezEdisdW), or download raw data from [ Bigscity-LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/)


