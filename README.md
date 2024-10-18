# BZK-InformationExtraction
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
