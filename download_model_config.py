from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForPreTraining,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
    AutoTokenizer
)


model_configs = {
    "gpt3_1_3B": {
        "model_name": "gpt2", 
        "config_args": {
            "use_cache": False,
            "remove_unused_columns": False,
            "return_dict": False,
            "n_head": 32,
            "num_hidden_layers": 24,
            "n_embd": 2048,
        }
    },
    "gpt3_2_7B": {
        "model_name": "gpt2", 
        "config_args": {
            "use_cache": False,
            "remove_unused_columns": False,
            "return_dict": False,
            "n_head": 32,
            "num_hidden_layers": 32,
            "n_embd": 2560,
        }
    },
    "gpt3_6_7B": {
        "model_name": "gpt2", 
        "config_args": {
            "use_cache": False,
            "remove_unused_columns": False,
            "return_dict": False,
            "n_head": 32,
            "num_hidden_layers": 32,
            "n_embd": 4096,
        }
    },
    "bert_340M": {
        "model_name": "bert-base-cased",
        "config_args": {
            "use_cache": False,
            "remove_unused_columns": False,
            "return_dict": False,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        }
    }
}

def download_model_config():

    for model_tag, config in model_configs.items():
        model_name = config["model_name"]
        config_args = config["config_args"]
        model_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name, **config_args
        )
        save_path = f"/workspace/Oobleck/data/model/{model_tag}"
        model_config.save_pretrained(save_path)

def download_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("/workspace/Oobleck/data/tokenizer/bert")
if __name__ == "__main__":
    pass
    download_model_config()
    # download_tokenizer()
