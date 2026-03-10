#!/bin/bash
# CineBrain虚拟环境激活脚本

# 清除代理环境变量（防止镜像源冲突）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活虚拟环境
source "$SCRIPT_DIR/venv/bin/activate"

echo "✅ CineBrain虚拟环境已激活"
echo "Python版本: $(python --version)"
echo "当前环境: $VIRTUAL_ENV"
echo ""
echo "验证关键包:"
python -c "import torch; print('  - PyTorch:', torch.__version__)"
python -c "import numpy; print('  - NumPy:', numpy.__version__)"
python -c "import datasets; print('  - datasets:', datasets.__version__)"
python -c "import pyarrow; print('  - pyarrow:', pyarrow.__version__)"
echo ""
echo "使用方法: source activate_venv.sh"
