import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#  设置 max_split_size_mb 环境变量来减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import configparser
import logging
import json

from pathlib import Path
from experiment import run
from random_seed import setup_seed
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




def parse_configurations(config_file: Path):

    config = configparser.ConfigParser()
    config.read(config_file)

    def get_option(section, option, default=None, type_func=str):
        if config.has_option(section, option):
            value = config.get(section, option)
            return type_func(value)
        return default



    def get_path(section, option, default=None, dtype=str):
        value = get_option(section, option, default, dtype)
        if value is None:
            return None  # 直接返回 None，而不是传给 Path()
        return Path(value).expanduser()

    def get_bool(section, option, default=False):
        return get_option(section, option, default, lambda v: v.lower() in ['true', '1'])

    def get_rgb(section, option, default=None):
        value = get_option(section, option, default, str)
        if value:
            return list(map(int, value.split(',')))
        return default

    # Read and process configuration options
    config_data = {
        'experiment_name': get_option('General', 'experiment_name', 'exp1'),
        'steps': get_option('General', 'steps', 'train,prediction,evaluation', lambda v: v.split(',')),
        'classes_names': get_option('General', 'classes_names', ' Fluid,Good,Dry,Tearing', lambda v: v.split(',')),
        'img_size': get_option('General', 'img_size', 384, int),
        'no_of_epochs': get_option('General', 'no_of_epochs', 100, int),
        'num_workers': get_option('General', 'num_workers', 0, int),
        'batch_size': get_option('General', 'batch_size', 4, int),
        'desired_batchsize': get_option('General', 'desired_batchsize', 4, int),
        'bin_size': get_option('General', 'bin_size', 20, int),
        'learning_rate': get_option('General', 'learning_rate', 5e-3, float),
        'use_amp': get_bool('General', 'use_amp', False),
        'model_path': get_path('Paths', 'model_path'),
        'prediction_path': get_path('Paths', 'prediction_path', 'prediction'),
        'evaluation_path': get_path('Paths', 'evaluation_path', 'evaluation'),
        'tb_path': get_path('Paths', 'tb_path', 'events'),
        'log_path': get_path('Paths', 'log_path', 'logs'),
        'loss':get_option('General','loss','initial'),
        'same_classes':get_bool('General','same_classes',False),
        'use_gpu': get_bool('General', 'use_gpu', True),
        'forbid_augmentations':get_bool('General', 'forbid_augmentations', False),
        'model_name': get_option('General', 'model_name', 'None', str),
        'seed': get_option('General', 'seed', 35, int),
        'norm_params': get_path('Paths', 'norm_params'),
        'extra_test_data': get_bool('General', 'extra_test_data', False),
        'prefetch_factor': get_option('General', 'prefetch_factor', 2, int),
        'label_smooth': get_option('General', 'label_smooth', 0.1, float),
        'weight_decay': get_option('General', 'weight_decay', 5e-3, float),
        'use_images_generees': get_bool('General', 'use_images_generees', False),
        'images_generees_path': get_path('Paths', 'images_generees_path'),
        'mean_features': get_path('Paths', 'mean_features'),
        'std_features': get_path('Paths', 'std_features'),
        'num_genere': get_option('General', 'num_genere', ' 0,0,0,0,0', lambda v: v.split(',')),
    }

    log_path = config_data['log_path'] /config_data['experiment_name']
    if len(config_data['steps'])==1 and config_data['steps'][0]=='evaluation':
        config_data["log_path"] = log_path
    else:
        index = 1
        original_log_path = log_path
        while log_path.exists():
            log_path = original_log_path.with_name(f"{original_log_path.name}_{index}")
            index += 1
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        config_data["log_path"]=log_path

    data_paths = {
        'train': {
            'image': get_path('DataPaths', 'train_image', '../data/concret_windows/train'),
        },
        'val': {
            'image': get_path('DataPaths', 'val_image', '../data/concret_windows/val'),

        },
        'test': {
            'image': get_path('DataPaths', 'test_image', '../data/concret_windows/test'),

        },
        'label_csv':{
            'label': get_path('DataPaths', 'label_csv', '../data/concret_windows/texture_windows-labels.csv'),
        },

    }
    config_data['data_paths'] = data_paths
    return config_data

def save_configuration(config: dict):


    os.makedirs(config["log_path"], exist_ok=True)
    path = Path(config["log_path"]) / f"{config['experiment_name']}.json"
    with open(path, "w") as config_file:
        json.dump(config, config_file, indent=4, default=str, sort_keys=True)
        logger.info(f"Saved configuration in {path.resolve()}")

def main():
    config_file = Path('./configuration.ini')  # Replace with your config file path
    config = parse_configurations(config_file)

    # Ensure we have at least one step to run
    if len(config["steps"]) == 0:
        logger.error("No step to run, exiting execution.")
        return

    # Save configuration to be able to re-run the experiment
    save_configuration(config)
    num_workers=config["num_workers"]
    setup_seed(config["seed"])
    # Run experiment
    run(config,num_workers)

if __name__ == "__main__":
    main()
