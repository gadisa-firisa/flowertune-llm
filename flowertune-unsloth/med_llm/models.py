"""flowertune: A Flower / FlowerTune app."""

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays
from unsloth import FastLanguageModel
import os

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    access_token = os.getenv("HF_ACCESS_TOKEN")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.name,
        load_in_4bit=model_cfg.quantization == 4,
        max_seq_length = 512,
        dtype = None,
        token = access_token
    )
    model = FastLanguageModel.get_peft_model(

        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = model_cfg.gradient_checkpointing,
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )
    
    return model, tokenizer


def set_parameters(model, parameters):
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def get_parameters(model):
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
