"""flowertune: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from med_llm.models import get_model
FDS = None  # Cache FederatedDataset

def formatting_prompts_func(examples, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {}
    ### Response:
    {}"""
    outputs = examples["response"]
    instructions = examples["instruction"]
    output_texts = []
    for instruct, output in zip(instructions, outputs):
        # Add EOS token to ensure proper sequence termination
        text = alpaca_prompt.format(instruct, output) + tokenizer.eos_token
        output_texts.append(text)
    return {"text": output_texts, }

def formatting(dataset):
    """Format dataset."""
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets."""
    dataset = dataset.rename_column("output", "response")
    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])
    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_name: str, tokenizer):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = reformat(client_trainset, llm_task="medical")
    return client_trainset.map(lambda examples: formatting_prompts_func(examples, tokenizer), batched=True)



def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
