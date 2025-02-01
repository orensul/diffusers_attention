import os

import argparse
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image  # Import the Pillow library
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffusers import FluxPipeline
from visualization_utils import visualize_tokens_attentions
from FLUX.flux_pipeline import AttentionFluxPipeline
from FLUX.flux_transformer import FluxTransformer2DModel
from huggingface_hub import login

# Hugging Face access tokens
access_token_read = "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"
access_token_write = "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"
login(token=access_token_write)
PROMPT_LEN = 77
prompts = ["a photo of an astronaut riding a horse on mars", "a photo of an astronaut sitting with a pina colada cocktail in the beach of Thiland", "a photo of an astronaut landing on mars", "a photo of an astronaut leaving the earth", "a photo of an astronaut with a cat on mars"]

# FLUX Model class
class FLUXModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def get_pipe(self):
        self.transformer  = FluxTransformer2DModel.from_pretrained(self.model_name, subfolder="transformer", torch_dtype=torch.float16)
        self.pipe =  AttentionFluxPipeline.from_pretrained(self.model_name, transformer=self.transformer, torch_dtype=torch.float16).to(self.device)
        return self.pipe

    def get_images(self, pipe, prompt, seed, n_steps, guidance_scale, height, width, prompt_length, perform_sdsa):
        query_store_kwargs = {'t_range': [0, n_steps // 10], 'strength_start': 0.9, 'strength_end': 0.81836735}

        if perform_sdsa:
            extended_attn_kwargs = {'t_range': [(5, n_steps - 10)]}
        else:
            extended_attn_kwargs = {'t_range': []}
        print(extended_attn_kwargs['t_range'])
        return pipe(
            prompt=prompt,
            generator=torch.Generator(self.device).manual_seed(seed),
            extended_attn_kwargs=extended_attn_kwargs,
            query_store_kwargs=query_store_kwargs,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=prompt_length,
        ).images


def show_heatmap(pipe, image, prompt):
    agg_attn = pipe.attention_store.aggregate_attention().float().cpu()
    tokens_text = [pipe.tokenizer_2.decode(x) for x in pipe.tokenizer_2(prompt, padding="max_length", return_tensors="pt").input_ids[0]]
    tokens_text = [f"{x}_{i}" for i, x in enumerate(tokens_text)]
    idx_range = (0, 20)
    visualize_tokens_attentions(agg_attn[0, idx_range[0]:idx_range[1]], tokens_text[idx_range[0]:idx_range[1]], image, prompt, heatmap_interpolation="bilinear")

def show_distribution(pipe, prompt):
    agg_attn = pipe.attention_store.aggregate_attention().float().cpu()
    print(agg_attn.shape)
    sum_attn_per_token = agg_attn.view(PROMPT_LEN, -1).mean(dim=1).cpu()
    sum_attn_per_token = sum_attn_per_token / sum_attn_per_token.sum()
    tokens_text = [pipe.tokenizer_2.decode(x) for x in pipe.tokenizer_2(prompt, padding="max_length", return_tensors="pt").input_ids[0]][:PROMPT_LEN]

    # Show a bar plot of the attention per token
    attn_per_token = {f'{t}_{i}': sum_attn_per_token[i] for i, t in enumerate(tokens_text)}

    plt.figure(figsize=(100, 30))
    plt.bar(attn_per_token.keys(), attn_per_token.values())
    plt.xticks(rotation=90)
    output_path = prompt + "_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    # plt.show()

def show_image(img, prompt):
    output_path = prompt + ".png"
    img.save(output_path)
    print(f"Image saved to {output_path}")



def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping



def test_model():
    flux_model = FLUXModel("black-forest-labs/FLUX.1-dev")
    pipe = flux_model.get_pipe()
    images = flux_model.get_images(
        pipe=pipe,
        prompt=prompts,
        seed=2,
        n_steps=30,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        prompt_length=PROMPT_LEN,
        perform_sdsa = True
    )
    for i in range(len(images)):
        show_image(images[i], prompts[i])

    # show_heatmap(pipe, img, prompt)
    # show_distribution(pipe, prompt)


def run_batch_generation(model, prompts, concept_token, seed=40, n_steps=50, mask_dropout=0.5, share_queries=True, perform_sdsa=True, downscale_rate=4, n_anchors=2):
    pipe = model.get_pipe()
    device = model.get_device()
    tokenizer = pipe.tokenizer_2
    float_type = pipe.dtype

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(n_anchors)))

    default_attention_store_kwargs = {'token_indices': token_indices, 'mask_dropout': mask_dropout, 'extended_mapping': anchor_mappings}


    return None, None



def run_batch(seed=40, mask_dropout=0.5, style="A photo of ", subject="a cute dog", concept_token=['dog'], settings=["sitting in the beach", "standing in the snow"], out_dir=None):
    flux_model = FLUXModel("black-forest-labs/FLUX.1-dev")
    prompts = [f'{style}{subject} {setting}' for setting in settings]
    images, image_all = run_batch_generation(flux_model, prompts, concept_token, seed, mask_dropout=mask_dropout)
    return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', default="batch", type=str, required=False)  # batch, cached
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--mask_dropout', default=0.5, type=float, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)
    parser.add_argument('--style', default="A photo of ", type=str, required=False)
    parser.add_argument('--subject', default="a cute dog", type=str, required=False)
    parser.add_argument('--concept_token', default=["dog"], type=str, nargs='*', required=False)
    parser.add_argument('--settings', default=["sitting on the bed", "sitting in the beach", "sitting on the desk"], type=str, nargs='*', required=False)
    parser.add_argument('--cache_cpu_offloading', default=False, type=bool, required=False)
    parser.add_argument('--out_dir', default="output_images", type=str, required=False)
    args = parser.parse_args()

    test_model()
