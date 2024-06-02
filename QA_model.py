import tensorflow as tf
from tensorflow.keras import layers, models
from einops import rearrange
import json
from .blocks import ModifiedResNet, PMC_CLIP_cfg
from transformers import CLIPVisionConfig
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
import numpy as np

@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config

class QA_model(tf.keras.Model):
    def __init__(self, model_args):
        super(QA_model, self).__init__()

        self.hidden_dim = model_args.hidden_dim
        self.voc_size = model_args.voc_size
        self.img_tokens = model_args.img_token_num
        self.H = model_args.H
        self.N = model_args.N
        self.Vision_module = model_args.Vision_module

        ###################################
        ''' Visual Model'''
        ###################################

        if self.Vision_module == 'PMC-CLIP':
            vision_cfg = PMC_CLIP_cfg()
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            vision_model = ModifiedResNet(
                layers=vision_cfg.layers,
                heads=vision_heads,
                output_dim=768,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
            vision_model = self.vision_load_pretrain(vision_model, model_args.visual_model_path)
            self.vision_model = models.Sequential(list(vision_model.layers)[:-2])
            num_ftrs = 1024
        if self.Vision_module == "CLIP":
            self.vision_model = tf.keras.applications.CLIPVisionModel(model_args.visual_model_path, include_top=False)
            num_ftrs = 768
        if self.Vision_module == 'Scratch':
            self.vision_model = tf.keras.applications.CLIPVisionModel(image_size=(512, 512), include_top=False)
            num_ftrs = 768

        ###################################
        ''' Query Decoder'''
        ###################################

        self.query_embed = layers.Embedding(self.img_tokens, num_ftrs)

        self.decoder_layer = ModifiedTransformerDecoderLayer(
            num_ftrs, self.H, 1024, 0.1, activation='relu', normalize_before=True)
        self.decoder = ModifiedTransformerDecoder(self.decoder_layer, self.N, normalize_before=True)

        ###################################
        ''' FC '''
        ###################################

        self.fc_l1 = layers.Dense(num_ftrs)
        self.fc_l2 = layers.Dense(self.hidden_dim)

        ###################################
        ''' Large Language Model'''
        ###################################

        self.llamacasual = self.Setup_model(model_args)

    def vision_load_pretrain(self, resnet, model_path):
        checkpoint = tf.train.Checkpoint(model=resnet)
        checkpoint.restore(model_path).expect_partial()
        return resnet

    def Setup_model(self, model_args):
        print("Setup Model")
        model = transformers.LlamaForCausalLM.from_pretrained(
           model_args.model_path,
        )
        if model_args.checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
        if model_args.is_lora:
            print("Setup PEFT")
            peft_config = get_peft_config(peft_args=model_args)
            model = get_peft_model(model, peft_config)
        return model

    def image_encoder(self, xis):
        if self.Vision_module == 'PMC-CLIP':
            batch_size = tf.shape(xis)[0]
            res_fea = self.vision_model(xis)
            out_emb = rearrange(res_fea, 'b d n1 n2 -> b (n1 n2) d')
        if self.Vision_module == 'CLIP' or self.Vision_module == 'Scratch':
            out_emb = self.vision_model(xis)[:, 1:, :]
        return out_emb

    def call(self, input_ids, images, labels=None):
        B = tf.shape(images)[0]

        x = self.image_encoder(images)
        features = tf.transpose(x, perm=[1, 0, 2])

