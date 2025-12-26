from copy import deepcopy
import os

_DEFAULT_CFG = {
    'GPU_ID': '0',
    "num_workers": 8,
    "random_seed": 42,
    "cfg_verbose": True,
    "save_dir": "../output/",
    "num_classes": 17,
    "width_mult": 1.0,
    "img_size": 192,
    'img_path': "../../dataset/imgs",
    'train_label_path': '../../dataset/train.json',
    'val_label_path': '../../dataset/val.json',
    'balance_data': False,
    'log_interval': 10,
    'pin_memory': True,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 120,
    'optimizer': 'Adam',
    'scheduler': 'MultiStepLR-70,100-0.1',
    'weight_decay': 5.e-4,
    'clip_gradient': 5,
    'acc_kps_th': 0.0,
    'log_batch_acc': False,
    'max_train_batches': 0,
    'use_data_aug': True,
    'resume_path': '',
    'resume_optimizer': True,
    'resume_scheduler': True,
    'resume_strict': True,
    'save_last': True,
    'early_stop_patience': 0,
    'early_stop_min_delta': 0.0,
    'test_img_path': "../../dataset/test/imgs",
    'eval_img_path': '../../dataset/eval/imgs',
    'eval_label_path': '../../dataset/eval/label.json',
    'eval_weights': '',
}

def _load_toml(path):
    if not os.path.exists(path):
        return {}

    try:
        import tomllib as _toml
    except ImportError:
        import tomli as _toml

    with open(path, 'rb') as f:
        data = _toml.load(f)

    if isinstance(data, dict) and isinstance(data.get('cfg'), dict):
        data = data['cfg']

    return data if isinstance(data, dict) else {}

def _coerce_value(value, default):
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'n', 'off'):
            return False
        return bool(value)

    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)

    if isinstance(default, float):
        return float(value)

    return str(value)

def _resolve_key(cfg, name):
    if name in cfg:
        return name
    lower = name.lower()
    if lower in cfg:
        return lower
    upper = name.upper()
    if upper in cfg:
        return upper
    return None

def _apply_mapping(cfg, mapping):
    for key, value in mapping.items():
        resolved = _resolve_key(cfg, key)
        if resolved is None:
            cfg[key] = value
            continue
        cfg[resolved] = _coerce_value(value, cfg[resolved])

def _apply_env(cfg, prefix='CFG_'):
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        key = env_key[len(prefix):]
        if not key:
            continue

        if '__' in key:
            parts = [part for part in key.split('__') if part]
            if not parts:
                continue
            current = cfg
            for part in parts[:-1]:
                resolved = _resolve_key(current, part) or part
                if resolved not in current or not isinstance(current[resolved], dict):
                    current[resolved] = {}
                current = current[resolved]

            last = parts[-1]
            resolved_last = _resolve_key(current, last) or last
            default_value = current.get(resolved_last)
            if default_value is None:
                current[resolved_last] = env_value
            else:
                current[resolved_last] = _coerce_value(env_value, default_value)
            continue

        resolved = _resolve_key(cfg, key)
        if resolved is None:
            cfg[key] = env_value
            continue
        cfg[resolved] = _coerce_value(env_value, cfg[resolved])

def load_config(path=None):
    cfg = deepcopy(_DEFAULT_CFG)
    cfg_file = path or os.environ.get('CFG_FILE') or 'config.toml'
    file_data = _load_toml(cfg_file)
    if file_data:
        _apply_mapping(cfg, file_data)
    _apply_env(cfg)
    return cfg
