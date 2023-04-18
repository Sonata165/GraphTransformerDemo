# Code for CS6208 assignment
## Reproduce Graph Transformer
Paper: https://arxiv.org/abs/2012.09699

Ou Longshen's assignment submission.

## Environment
    # On linux (CUDA 11.7)
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
    pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

## Files
    data            Dataset used in the experiment.
    models_gt.py    Code to define the Graph Transformer model.
    train.py        Code for model training and evaluation.
    train.ipynb     Training log of one experiment.

## Run the code
Check the train.ipynb for previous training logs. 

Or use below command:
    
    python train.py
to run the training script. 