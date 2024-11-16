import warnings
import os
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_contour

warnings.filterwarnings('ignore')

import sys

sys.path.append('.')
sys.path.append('..')
import yaml
import argparse
import traceback
import time
import torch

from model.models import D2STCNet
from model.trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph
)


def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(args.device)

    ## load dataset
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        time_slot=args.input_length,
        time_of_day=args.time_of_day,
    )
    if hasattr(args, 'graph_file'):
        graph = load_graph(args.graph_file, device=args.device)
    else:
        graph = None
    # args.num_nodes = len(graph)

    ## init model and set optimizer
    model = D2STCNet(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters,
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )

    ## start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        args=args,
        # graph=graph,
    )
    results = None
    try:
        if args.mode == 'train':
            results = trainer.train()  # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'], trainer.logger, trainer.args, graph)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    model_supervisor(args)


def objective(trial):
    # Dynamically set hyperparameters based on the trial
    # args.d_model = trial.suggest_categorical("d_model", [32, 48, 64])
    args.lr_init = trial.suggest_float("lr_init", 1e-3, 2e-3, log=True)
    args.dropout = trial.suggest_float("dropout", 0.1, 0.4)
    args.percent = trial.suggest_float("percent", 0.1, 0.3)
    # args.shm_temp = trial.suggest_float("shm_temp", 0.1, 1.0)
    args.nmb_prototype = trial.suggest_int("nmb_prototype", 2, 10)
    # args.yita = trial.suggest_float("yita", 0.4, 0.6)
    args.time_of_day = trial.suggest_categorical("time_slot", [12, 24, 48, 60])
    # args.theta = trial.suggest_float("theta", 0.1, 0.5)

    # More parameters can be added for tuning

    # Call the previously defined model_supervisor function to start training
    results = model_supervisor(args)

    # Assuming results returns a dictionary containing 'val_loss' key with the validation loss
    val_loss = results['best_val_loss']
    return val_loss



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
#                         type=str, help='the configuration to use')
#
#     args = parser.parse_args()
#
#     # 创建一个Optuna研究对象
#     study = optuna.create_study(direction="minimize")
#
#     # 加载基础配置参数
#     base_configs = yaml.load(open(args.config_filename), Loader=yaml.FullLoader)
#     args = argparse.Namespace(**base_configs)
#
#     # 执行优化
#     study.optimize(objective, n_trials=100)  # 指定尝试的次数
#
#     print("Number of finished trials: ", len(study.trials))
#     print("Best trial:")
#     trial = study.best_trial
#
#     print("Value: ", trial.value)
#     print("Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
#
#     try:
#         ax1 = plot_optimization_history(study)
#         ax1.figure.savefig('optimization_history.png')
#
#         ax2 = plot_param_importances(study)
#         ax2.figure.savefig('param_importances.png')
#
#     except Exception as e:
#         print(f"An error occurred while generating visualizations: {e}")