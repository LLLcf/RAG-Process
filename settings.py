import yaml
import os
import torch

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 自动处理设备选择
    if config['system']['device'] == "auto":
        config['system']['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    elif config['system']['device'] == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Fallback to CPU.")
        config['system']['device'] = "cpu"
        
    return config

# 单例模式，其他文件直接 import CONF 即可
CONF = load_config()