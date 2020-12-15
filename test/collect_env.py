# This script outputs relevant system environment info
# Run it with `python collect_env.py`.
import os
import platform
import re
import torch

LEVEL_OFFSET = '\t'
KEY_PADDING = 20


def run_and_parse_first_match(run_lambda, command, regex):
    """
    Runs command using run_lambda, returns the first regex match if it exists
    """
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', r'V(.*)$')


def info_system():
    return {
        'OS': platform.system(),
        'architecture': platform.architecture(),
        'version': platform.version(),
        'processor': platform.processor(),
        'python': platform.python_version(),
    }


def info_cuda():
    info = {}
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    else:
        info['cuda'] = "not found"
    return info


def info_packages():
    return {
        "torch_version": torch.__version__
    }


def nice_print(details, level=0):
    lines = []
    for k in sorted(details):
        key = f'* {k}:' if level == 0 else f'- {k}:'
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level+1) * LEVEL_OFFSET + '- ' + v for v in details[k]]
        else:
            template = '{:%is} {}' % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def main():
    details = {
        "System": info_system(),
        'CUDA': info_cuda(),
        'Packages': info_packages(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)


if __name__ == '__main__':
    main()
