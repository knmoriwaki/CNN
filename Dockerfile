# PyTorch の公式イメージ（CUDA対応）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 作業ディレクトリを設定
WORKDIR /mnt/data/moriwaki/CNN

# Python の基本パッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Notebook のポート解放
EXPOSE 8888

# `main.py` をコンテナにコピー（必要なら）
COPY main.py .

# 実行コマンド
CMD ["python", "main.py"]

