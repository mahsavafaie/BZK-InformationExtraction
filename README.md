# Inferable (INFormation ExtRAction BundeszentraLkartEi)
For collaborative work on extracting information from BZK index cards


## Environment Setup

```
conda env create -f environment.yml
conda activate inferable
pip install -r requirements.txt
```

## Inference

to run on a directory that contains images with one of the InternvlModels (the default is InternVL2-Llama3-76B)

```
python main.py -i <address to the directory> -m class=InternvlModel -g <gpu specification>
```

To choose a different InternVL2 model, change model_name in the __init__ function in /inferable/models/Internvl_model.py


## Quantisation

To activate quantisation to use less memory, uncomment #load_in_8bit=True in line 141 of /inferable/models/Internvl_model.py



# Fine-tune InternVl
Install their library (further information can be found at their [installation website](https://internvl.readthedocs.io/en/latest/get_started/installation.html))
- First clone their repository:
```
git clone https://github.com/OpenGVLab/InternVL.git
``` 
- Create a conda virtual environment and activate it:
```
conda create -n internvl python=3.9 -y
conda activate internvl
```
- Install dependencies using requirements.txt:
```
pip install -r requirements.txt
```
- Install flash-attn==2.3.6:
```
pip install flash-attn==2.3.6 --no-build-isolation
```

#### Further modifications
- Due to error `cannot import name 'log' from 'torch.distributed.elastic.agent.server.api` an [github issue](https://github.com/huggingface/alignment-handbook/issues/180) proposed to update `deepspeed`:
```
pip install deepspeed==0.14.4
```
- Due to error `No module named 'datasets'`, install datasets:
```
pip install datasets==2.16.1
```
- Due to error that cuda is not found (TODO: check if still required):
```
conda install pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-cuda=12.4 -c pytorch -c nvidia
```
- Due to missing CUDA (`CUDA_SETUP: WARNING! libcudart.so not found in any environmental path.`), update bitsandbytes:
```
pip install bitsandbytes==0.44.1
```
Then also the command `python -m bitsandbytes` works fine.
- Due to error:
```
[rank0]:   File "/miniconda3/envs/internvl/lib/python3.9/site-packages/transformers/trainer.py", line 3979, in create_accelerator_and_postprocess
[rank0]:     self.accelerator = Accelerator(
[rank0]: TypeError: __init__() got an unexpected keyword argument 'dispatch_batches'
```
install compatible version of accelerate:
```
pip install accelerate==0.28.0
```


#### Download their models
- Go to folder `internvl_chat` and execute:
```
mkdir pretrained
cd pretrained/
```
- Depending on the model, execute:
```
# Download OpenGVLab/InternVL2-1B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B
# Download OpenGVLab/InternVL2-2B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
# Download OpenGVLab/InternVL2-4B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-4B --local-dir InternVL2-4B
# Download OpenGVLab/InternVL2-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
# Download OpenGVLab/InternVL2-26B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-26B --local-dir InternVL2-26B
# Download OpenGVLab/InternVL2-40B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-40B --local-dir InternVL2-40B
# Download OpenGVLab/InternVL2-Llama3-76B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-Llama3-76B --local-dir InternVL2-Llama3-76B
```

#### Write training data
To write the training data, execute:
```
python main.py -d bzk_small_raw -m class=InternvlWriteTraining,root_folder={pointing absolute to InternVL/internvl_chat}
```
and point the `root_folder` to the folder in the cloned internvl directory `InternVL/internvl_chat`.

#### Run training

- Execute (depending on the model) the corresponding script (check that you are still in folder `internvl_chat` in the cloned repository)