import os

import argparse
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image  # Import the Pillow library
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

import yaml
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
prompts_path = 'prompts_benchmark/prompts.yaml'
# prompts = ["a photo of an astronaut riding a horse on mars", "a photo of an astronaut sitting with a pina colada cocktail in the beach of Thiland", "a photo of an astronaut landing on mars", "a photo of an astronaut leaving the earth", "a photo of an astronaut with a cat on mars"]
# prompts = ["A photo of An athletic woman, lifting weights", "A photo of An athletic woman, at a sports stadium", "A photo of An athletic woman, wearing a tracksuit", "A photo of An athletic woman, dressed in a ballet outfit", "A photo of An athletic woman, wearing hiking attire"]


prompts1 =  [
                "A fairy, in a mystical glen",
                "A fairy, dressed in a petal gown",
                "A fairy, inside a hollowed-out tree",
                "A fairy, collecting morning dew",
                "A fairy, under a full moon"
            ]
prompts2 = [
                "A photo of A happy hedgehog, eating a big piece of cheese",
                "A photo of A happy hedgehog, in a garden",
                "A photo of A happy hedgehog, being held by a caretaker",
                "A photo of A happy hedgehog, dressed in a festive outfit",
                "A photo of A happy hedgehog, in an autumn forest"
            ]

prompts3 = [
                "A photo of A dog, chasing a frisbee",
                "A photo of A dog, on a beach",
                "A photo of A dog, sleeping on a porch",
                "A photo of A dog, dressed in a superhero cape",
                "A photo of A dog, sitting by a fireplace"
            ]

prompts4 = [
                "A watercolor illustration of A puppy, wearing a small sweater",
                "A watercolor illustration of A puppy, digging a hole",
                "A watercolor illustration of A puppy, wearing a training harness",
                "A watercolor illustration of A puppy, sticking head out of the car window",
                "A watercolor illustration of A puppy, swimming"
            ]

prompts5 = ["neonpunk style of A parrot, in a tropical rainforest",
"neonpunk style of A parrot, dressed in a colorful necklace",
"neonpunk style of A parrot, interacting with a mirror",
"neonpunk style of A parrot, sitting on a pirate's shoulder",
"neonpunk style of A parrot, in a jungle"]


prompts6 = [
                "A hyper-realistic digital painting of A middle-aged man, wearing a business suit",
                "A hyper-realistic digital painting of A middle-aged man, at a traditional Japanese garden",
                "A hyper-realistic digital painting of A middle-aged man, in a mountain cabin",
                "A hyper-realistic digital painting of A middle-aged man, walking a dog",
                "A hyper-realistic digital painting of A middle-aged man, wearing a tuxedo"
            ]

prompts7 = [
                "origami style of A dragon, atop an ancient castle",
                "origami style of A dragon, breathing a plume of fire",
                "origami style of A dragon, coiling around its hoard",
                "origami style of A dragon, guarding a treasure trove",
                "origami style of A dragon, resting in a cavern"
            ]

prompts = [prompts1, prompts2, prompts3, prompts4, prompts5, prompts6, prompts7]




# end range should be at most 17

timestep_ranges = [[3, 8], [3, 11], [3, 14], [5, 17], [5, 14], [5, 11], [7, 10], [7, 13], [7, 16]]
timestep_ranges = [[5, 15]]


layers_config = ["none", "single_even", "multi", "single", "quarter1", "quarter2", "quarter3", "quarter4", "mix", "multi_even"]


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

    def get_images(self, pipe, prompt, seed, n_steps, guidance_scale, height, width, prompt_length, perform_sdsa, timestep_start_range, timestep_end_range, layers_extended_config):
        query_store_kwargs = {'t_range': [0, n_steps // 10], 'strength_start': 0.9, 'strength_end': 0.81836735}

        if perform_sdsa:
            extended_attn_kwargs = {'t_range': [(timestep_start_range, timestep_end_range)]}
        else:
            extended_attn_kwargs = {'t_range': []}
        print(extended_attn_kwargs['t_range'])
        return pipe(
            prompt=prompt,
            generator=torch.Generator(self.device).manual_seed(seed),
            extended_attn_kwargs=extended_attn_kwargs,
            query_store_kwargs=query_store_kwargs,
            layers_extended_config=layers_extended_config,
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



def test_model(args):
    flux_model = FLUXModel("black-forest-labs/FLUX.1-dev")
    pipe = flux_model.get_pipe()
    for timestep_start, timestep_end in timestep_ranges:
        for prompts_in_batch in prompts:
            for layer_conf in layers_config:
                images = flux_model.get_images(
                    pipe=pipe,
                    prompt=prompts_in_batch,
                    seed=2,
                    n_steps=30,
                    guidance_scale=3.5,
                    height=1024,
                    width=1024,
                    prompt_length=PROMPT_LEN,
                    perform_sdsa = True,
                    timestep_start_range=timestep_start,
                    timestep_end_range=timestep_end,
                    layers_extended_config=layer_conf
                )
                for i in range(len(images)):
                    prompt = prompts_in_batch[i]
                    img = images[i]
                    prompt = prompt.replace(" ", "_")
                    output_path = f"{prompt}_timestep_{timestep_start}_{timestep_end}_layers_config_{layer_conf}.png"
                    img.save(output_path)
                    print(f"Image saved to {output_path}")




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


def read_prompts():
    with open(prompts_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    prompts = []
    for category, items in data.items():
        print(f"\nCategory: {category}")
        for item in items:
            print(f"  Subject: {item['subject']}")
            for prompt in item["prompts"]:
                print(f"    - {prompt}")
                prompts.append((category, item['subject'], prompt))
    return prompts


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

    # prompts = read_prompts()[:5]
    # prompts = [prompt for _, _, prompt in prompts]
    test_model(args)
