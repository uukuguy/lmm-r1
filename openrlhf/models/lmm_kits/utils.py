from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.configuration_utils import PretrainedConfig
import importlib
import os


def _get_kit_root_path(pretrain_or_model=None,model_type=None):
    assert (pretrain_or_model is not None) ^ (model_type is not None), "only and only one of pretrain_or_model and model_type should be provided"
    if model_type is None:
        # Safely load config without excuting remote code.
        config = PretrainedConfig.from_pretrained(pretrain_or_model)
        model_type = config.model_type
    root_path = f".models.lmm_kits.{model_type}"
    #check if the module exists.
    if not importlib.util.find_spec(root_path,package="openrlhf"):
        # This is an llm or unsupported lmm.
        root_path = f".models.lmm_kits.llm"
    return root_path

def _get_hf_processor(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    processor_kwargs = strategy.args.processor_kwargs
    # There maybe some patches for the processor
    load_patch(pretrain_or_model=pretrain)
    processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=False, use_fast=use_fast, **processor_kwargs)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor

def get_data_processor(pretrain_or_model, model, padding_side="left", strategy=None, use_fast=True):
    root_path = _get_kit_root_path(pretrain_or_model)
    module = importlib.import_module(f"{root_path}.data_processor",package="openrlhf")
    data_processor_cls = getattr(module, "DataProcessor")
    hf_processor = _get_hf_processor(pretrain_or_model, model, padding_side, strategy,use_fast=use_fast)
    data_processor = data_processor_cls(hf_processor,processor_kwargs=strategy.args.processor_kwargs)
    return data_processor

def load_patch(pretrain_or_model=None,model_type=None, use_liger_kernel=False):
    # only and only one of pretrain_or_model and model_type should be provided
    # use xor to check
    assert (pretrain_or_model is not None) ^ (model_type is not None), "only and only one of pretrain_or_model and model_type should be provided"
    root_path = _get_kit_root_path(pretrain_or_model,model_type)
    module = importlib.import_module(f"{root_path}.patch",package="openrlhf")
    Patch = getattr(module, "Patch")
    Patch.load_all_patches(use_liger_kernel=use_liger_kernel)

def get_generation_cls(pretrain_or_model, use_liger_kernel=False):
    model_type = PretrainedConfig.from_pretrained(pretrain_or_model).model_type
    load_patch(model_type=model_type, use_liger_kernel=use_liger_kernel)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if model_type in CONFIG_MAPPING:
        # load_patch register customized config if needed, so we ensure AutoConfig loads our customized config, not remote config.
        # This also ensure type(config) mapped to our customized model class.
        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=False)
    else:
        # Not a model customized by lmm-r1 or supported by transformers, so we trust remote code.
        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    if type(config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText._model_mapping[type(config)]
    else:
        # Compatible with LLMs
        if use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            return AutoLigerKernelForCausalLM
        else:
            return AutoModelForCausalLM

def hack_peft_model(peft_model):
    def get_inputs_embeds(*args,**kwargs):
        return peft_model.base_model.model.get_inputs_embeds(*args,**kwargs)
    def get_position_ids(*args,**kwargs):
        return peft_model.base_model.model.get_position_ids(*args,**kwargs)
    def offset_split_position_ids(*args,**kwargs):
        return peft_model.base_model.model.offset_split_position_ids(*args,**kwargs)
    peft_model.get_inputs_embeds = get_inputs_embeds
    peft_model.get_position_ids = get_position_ids
    peft_model.offset_split_position_ids = offset_split_position_ids
    return peft_model
