```bash
sudo docker build -t llmf .
```

```bash
git clone https://github.com/mosaicml/llm-foundry.git
```

```bash
sudo docker run --rm --gpus all --shm-size 100gb --runtime nvidia -v $PWD/data:/data -v $PWD/experiments:/experiments -v $PWD/llm-foundry:/llm-foundry -e HF_HOME="/data" -it llmf
```

```bash
pip install -e ./llm-foundry
```

If model not downloaded, download it:
```bash
python3 experiments/edit_and_save_model.py
```

Launch
```bash
composer experiments/train.py experiments/yamls/run0.yaml
```