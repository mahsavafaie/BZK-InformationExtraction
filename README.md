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
- Change to the root folder of the cloned repository:
```
cd InternVL
```
- Create a conda virtual environment and activate it:
```
conda create -n internvl python=3.9 -y
conda activate internvl
```
- Install dependencies using requirements.txt:
```
pip install -r requirements/internvl_chat.txt
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
- Go to folder `InternVL/internvl_chat` and execute:
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

- Execute (depending on the model) [the corresponding script](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html#start-2nd-fine-tuning) (check that you are still in folder `internvl_chat` in the cloned repository)
    - e.g. for fine-tuning the 1B model:
```# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
```



#### In Case of LORA, Merge Weights

Based on [issue 468](https://github.com/OpenGVLab/InternVL/issues/468#issuecomment-2353846695) the weights of the lora model needs to be merged after training.

```
python tools/merge_lora.py {directory of finetuned model} {directory of merged model}
```

Most of the time, the train models resides at `work_dirs/internvl_chat_v2_0`, thus a command can look like this (for the 40B version):

```
python tools/merge_lora.py work_dirs/internvl_chat_v2_0/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_lora work_dirs/internvl_chat_v2_0/40bmerged
```

(For me `python tools/merge_lora.py`didn't work because it does not find the internvl folder. Thus I copied the `merge_lora.py` one folder up to `/internvl_chat` and then executed `python merge_lora.py` with the additonal arguments)

#### In All Cases, Copy Python Files

The architecture files are not copied during training, thus copy all python files from the original model (e.g. `pretrained/InternVL2-40B/*.py`) to the final folder.
A command can look like this:

```
cp {directory of pretrained model}/*.py {directory of finetuned/merged model}
```

For a lora 40B model the command looks like:
```
cp pretrained/InternVL2-40B/*.py work_dirs/internvl_chat_v2_0/40bmerged
```