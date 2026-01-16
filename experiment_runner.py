import traceback
import sys
import os

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config_dev", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)

    try:
        # Lazy import to facilitate faster Hydra config loading
        from src.experiment import Experiment

        experiment = Experiment(config)
        experiment.run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
