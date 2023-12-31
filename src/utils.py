import os
import random as rd
from datetime import datetime as dt
from typing import Dict, Tuple, Union

import torch
from omegaconf import OmegaConf
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Text2TextGenerationPipeline,
    TextGenerationPipeline,
    Trainer,
)


def set_seed(seed: int):
    rd.seed(42)
    torch.manual_seed(seed)


def is_seq2seq(inputs: Union[PreTrainedModel, PretrainedConfig, PreTrainedTokenizerBase]) -> bool:
    """
    return: bool
            true is Seq2Seq Model, false is Causal LM
    """
    if isinstance(inputs, PreTrainedTokenizerBase):
        inputs = AutoConfig.from_pretrained(inputs.name_or_path)

    if hasattr(inputs, "config"):
        inputs = inputs.config

    if not hasattr(inputs, "is_encoder_decoder"):
        raise TypeError(
            "type of inputs must be PreTrainedModel or PretrainedConfig.\n"
            f"type of input: {type(inputs)}"
        )

    return inputs.is_encoder_decoder


def prepare_model_conf(cfg: OmegaConf) -> Dict:
    cfg = OmegaConf.to_object(cfg)

    if cfg.get("torch_dtype") is not None:
        cfg["torch_dtype"] = getattr(torch, cfg["torch_dtype"])

    if cfg.get("device_map") == "local_rank":
        cfg["device_map"] = int(os.environ.get("LOCAL_RANK", 0))

    if "quantization_config" in cfg:
        quant_config = cfg["quantization_config"]
        if "compute_dtype" in quant_config:
            quant_config["compute_dtype"] = getattr(torch, quant_config["compute_dtype"])

        cfg["quantization_config"] = BitsAndBytesConfig(**quant_config)

    return cfg


def load_model_and_tokenizer(
    cfg: OmegaConf,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)

    model_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    model_cls = AutoModelForSeq2SeqLM if is_seq2seq(model_cfg) else AutoModelForCausalLM
    model = model_cls.from_pretrained(**prepare_model_conf(cfg.model))

    # lora
    lora_cfg = cfg.get("lora_config")
    if lora_cfg is not None:  # lora
        lora_cfg = OmegaConf.to_object(lora_cfg)
        lora_cfg["task_type"] = "SEQ_2_SEQ_LM" if is_seq2seq(model) else "CAUSAL_LM"
        peft_config = LoraConfig(**lora_cfg)

        model = get_peft_model(model, peft_config)

    return model, tokenizer


def load_pipeline(cfg: OmegaConf) -> Pipeline:
    is_lora = False
    try:  # peft 먼저 검색
        peft_cfg = PeftConfig.from_pretrained(cfg.pretrained_model_name_or_path)
        adapter_path = cfg.pretrained_model_name_or_path
        cfg.pretrained_model_name_or_path = peft_cfg.base_model_name_or_path
        is_lora = True
    except:
        pass

    try:  # transformers에서 검색
        model_cfg = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
    except Exception as e:
        raise e

    model_cls = AutoModelForSeq2SeqLM if is_seq2seq(model_cfg) else AutoModelForCausalLM
    model = model_cls.from_pretrained(**prepare_model_conf(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)

    if is_lora:
        model = PeftModel.from_pretrained(model, adapter_path)

    pipeline_cls = Text2TextGenerationPipeline if is_seq2seq(model) else TextGenerationPipeline
    pipe = pipeline_cls(model=model, tokenizer=tokenizer)

    return pipe


class TimeChecker:
    def __enter__(self):
        self.start = dt.now()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"time taken: {(dt.now() - self.start).total_seconds()}sec")
