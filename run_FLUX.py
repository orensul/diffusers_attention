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
from datetime import datetime

# Get today's date in YYYY-MM-DD format
today_date = datetime.today().strftime('%Y-%m-%d')

base_output_dir = "/cs/labs/dshahaf/orens/diffusers_attention/outputs"
output_dir = os.path.join(base_output_dir, today_date)
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists


# Hugging Face access tokens
access_token_read = "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"
access_token_write = "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"
login(token=access_token_write)
PROMPT_LEN = 512
prompts_path = 'prompts_benchmark/prompts.yaml'
# prompts = ["a photo of an astronaut riding a horse on mars", "a photo of an astronaut sitting with a pina colada cocktail in the beach of Thiland", "a photo of an astronaut landing on mars", "a photo of an astronaut leaving the earth", "a photo of an astronaut with a cat on mars"]
# prompts = ["A photo of An athletic woman, lifting weights", "A photo of An athletic woman, at a sports stadium", "A photo of An athletic woman, wearing a tracksuit", "A photo of An athletic woman, dressed in a ballet outfit", "A photo of An athletic woman, wearing hiking attire"]

seed = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

prompts8 = [
    "A photorealistic illustration of Kaylee a young woman in her early 20s with a slim yet athletic build. She has long, wavy brown hair that cascades past her shoulders, with a few loose strands framing her oval-shaped face. Her hazel eyes carry a warm yet focused expression, and her slightly arched eyebrows give her a naturally confident look. Her light skin has a subtle rosy undertone. She wears minimal makeup, with a touch of mascara and soft pink lips. She is dressed in a crisp white blouse, slightly loose-fitting but tucked neatly into her high-waisted black pants, which are tailored and elegant. The blouse has a V-neck with delicate buttons running down the front, and the sleeves are long with cuffs that are folded back slightly. She wears black leather loafers or modest heels, adding to her polished yet practical appearance. On her left wrist, she wears a silver watch, and a small silver pendant necklace rests just below her collarbone. Her posture is relaxed yet poised—standing with one hand lightly touching her hip while the other rests casually by her side. She holds a small leather handbag over her shoulder, its thin strap resting diagonally across her body. Kaylee reaches into her bag, pulling out a small notebook and pen before jotting something down.",
    "A photorealistic illustration of Kaylee a young woman in her early 20s with a slim yet athletic build. She has long, wavy brown hair that cascades past her shoulders, with a few loose strands framing her oval-shaped face. Her hazel eyes carry a warm yet focused expression, and her slightly arched eyebrows give her a naturally confident look. Her light skin has a subtle rosy undertone. She wears minimal makeup, with a touch of mascara and soft pink lips. She is dressed in a crisp white blouse, slightly loose-fitting but tucked neatly into her high-waisted black pants, which are tailored and elegant. The blouse has a V-neck with delicate buttons running down the front, and the sleeves are long with cuffs that are folded back slightly. She wears black leather loafers or modest heels, adding to her polished yet practical appearance. On her left wrist, she wears a silver watch, and a small silver pendant necklace rests just below her collarbone. Her posture is relaxed yet poised—standing with one hand lightly touching her hip while the other rests casually by her side. She holds a small leather handbag over her shoulder, its thin strap resting diagonally across her body. Kaylee leans against the wooden desk, one hand resting on the counter as she speaks with the hotel concierge.",
    "A photorealistic illustration of Kaylee a young woman in her early 20s with a slim yet athletic build. She has long, wavy brown hair that cascades past her shoulders, with a few loose strands framing her oval-shaped face. Her hazel eyes carry a warm yet focused expression, and her slightly arched eyebrows give her a naturally confident look. Her light skin has a subtle rosy undertone. She wears minimal makeup, with a touch of mascara and soft pink lips. She is dressed in a crisp white blouse, slightly loose-fitting but tucked neatly into her high-waisted black pants, which are tailored and elegant. The blouse has a V-neck with delicate buttons running down the front, and the sleeves are long with cuffs that are folded back slightly. She wears black leather loafers or modest heels, adding to her polished yet practical appearance. On her left wrist, she wears a silver watch, and a small silver pendant necklace rests just below her collarbone. Her posture is relaxed yet poised—standing with one hand lightly touching her hip while the other rests casually by her side. She holds a small leather handbag over her shoulder, its thin strap resting diagonally across her body. Kaylee checks the time on her silver watch, tapping her fingers against her wrist thoughtfully.",
    "A photorealistic illustration of Kaylee a young woman in her early 20s with a slim yet athletic build. She has long, wavy brown hair that cascades past her shoulders, with a few loose strands framing her oval-shaped face. Her hazel eyes carry a warm yet focused expression, and her slightly arched eyebrows give her a naturally confident look. Her light skin has a subtle rosy undertone. She wears minimal makeup, with a touch of mascara and soft pink lips. She is dressed in a crisp white blouse, slightly loose-fitting but tucked neatly into her high-waisted black pants, which are tailored and elegant. The blouse has a V-neck with delicate buttons running down the front, and the sleeves are long with cuffs that are folded back slightly. She wears black leather loafers or modest heels, adding to her polished yet practical appearance. On her left wrist, she wears a silver watch, and a small silver pendant necklace rests just below her collarbone. Her posture is relaxed yet poised—standing with one hand lightly touching her hip while the other rests casually by her side. She holds a small leather handbag over her shoulder, its thin strap resting diagonally across her body. Kaylee places her suitcase beside her, tapping her foot lightly as she waits for check-in.",
    "A photorealistic illustration of Kaylee a young woman in her early 20s with a slim yet athletic build. She has long, wavy brown hair that cascades past her shoulders, with a few loose strands framing her oval-shaped face. Her hazel eyes carry a warm yet focused expression, and her slightly arched eyebrows give her a naturally confident look. Her light skin has a subtle rosy undertone. She wears minimal makeup, with a touch of mascara and soft pink lips. She is dressed in a crisp white blouse, slightly loose-fitting but tucked neatly into her high-waisted black pants, which are tailored and elegant. The blouse has a V-neck with delicate buttons running down the front, and the sleeves are long with cuffs that are folded back slightly. She wears black leather loafers or modest heels, adding to her polished yet practical appearance. On her left wrist, she wears a silver watch, and a small silver pendant necklace rests just below her collarbone. Her posture is relaxed yet poised—standing with one hand lightly touching her hip while the other rests casually by her side. She holds a small leather handbag over her shoulder, its thin strap resting diagonally across her body. Kaylee presses the elevator button, watching the golden numbers above the door light up in sequence.",

]

prompts9 = [
    "a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee reaches into her bag, pulling out a small notebook and pen before jotting something down.",
    "a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee leans against the wooden desk, one hand resting on the counter as she speaks with the hotel concierge.",
    "a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee checks the time on her silver watch, tapping her fingers against her wrist thoughtfully.",
    "a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee places her suitcase beside her, tapping her foot lightly as she waits for check-in.",
    "a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee presses the elevator button, watching the golden numbers above the door light up in sequence."
]

prompts10 = [
    "A photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk",'a photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk, she places the clrealy visable sleek silver letter opener with an engraved handle, inside the wooden desk drawer, polished and dark brown,.',
    "A photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand in the  spacious hotel lobby with marble floors, a chandelier, and a wooden desk, she place the clrealy visable sleek silver letter opener with an engraved handle, inside the wooden desk drawer, polished and dark brown,, Liam, a young man with short black hair, wearing a blue shirt and dark jeans, watches her from far.",'a photorealistic illustration of The spacious hotel lobby with marble floors, a chandelier, and a wooden desk is empty.',
    "A photorealistic illustration of Liam, a young man with short black hair, wearing a blue shirt and dark jeans, stand in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk.",
    "A photorealistic illustration of Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, and Liam, a young man with short black hair, wearing a blue shirt and dark jeans, standing in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk.",
    "A photorealistic illustration of Liam, a young man with short black hair, wearing a blue shirt and dark jeans, moves the clrealy visable sleek silver letter opener with an engraved handle, to the classic leather briefcase with brass buckles, in the spacious hotel lobby with marble floors, a chandelier, and a wooden desk. Kaylee, a young woman with long brown hair, wearing a white blouse and black pants, stand there and watch him.",
    ]
prompts = [prompts1, prompts2, prompts3, prompts4, prompts5, prompts6, prompts7]
prompts = [prompts10]



# timestep_ranges = [[3, 8], [3, 11], [3, 14], [5, 17], [5, 14], [5, 11], [7, 10], [7, 13], [7, 16]]
# timestep_ranges = [[0, 20]]
# layers_config = ["none",  "multi", "multi_even",  "multi_first_half", "multi_second_half", "q1", "q2", "q3", "q4", "single", "single_even",  "single_first_half", "single_second_half", "mix"]
# layers_config = ["none", ""]

extended_attn_kwargs = {'t_range': [(0, 20)]}







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
        if perform_sdsa:
            extended_attn_kwargs = {'t_range': [(timestep_start_range, timestep_end_range)]}
        else:
            extended_attn_kwargs = {'t_range': []}

        return pipe(
            prompt=prompt,
            generator=torch.Generator(self.device).manual_seed(seed),
            extended_attn_kwargs=extended_attn_kwargs,
            layers_extended_config=layers_extended_config,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=prompt_length,
        ).images




def test_model(prompts):
    flux_model = FLUXModel("black-forest-labs/FLUX.1-dev")
    pipe = flux_model.get_pipe()
    for prompts_in_batch_id, prompts_in_batch in enumerate(prompts):
        for single_config, multi_config in [("none", "none"), ("even", "second_half"), ("first_half", "second_half")]: # , ("even", "second_half"), ("even", "first_half"), ("even", "even"), ("first_half", "second_half"), ("first_half", "first_half"), ("first_half", "even"), ("second_half", "second_half"), ("second_half", "first_half"), ("second_half", "even")]:
            for dropout in [0.0]:
                images = pipe(
                    prompt=prompts_in_batch,
                    guidance_scale=3.5,
                    height=1024,
                    width=1024,
                    num_inference_steps=30,
                    extended_attn_kwargs=extended_attn_kwargs,
                    layers_extended_config={'single': single_config, 'multi': multi_config},
                    generator=torch.Generator(device).manual_seed(seed),
                    max_sequence_length=PROMPT_LEN,
                    dropout_value=dropout,
                    same_latents=False,
                ).images
                for i in range(len(images)):
                    prompt = prompts_in_batch[i]
                    img = images[i]
                    output_path = os.path.join(output_dir, f"sample_id_{prompts_in_batch_id}_prompt_{i}_timestep_{extended_attn_kwargs['t_range'][0][0]}_{extended_attn_kwargs['t_range'][0][1]}_layers_config_{single_config}_{multi_config}.png")
                    img.save(output_path)
                    print(f"Image saved to {output_path}")



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



# def run_batch_generation(model, prompts, concept_token, seed=40, n_steps=50, mask_dropout=0.5, share_queries=True, perform_sdsa=True, downscale_rate=4, n_anchors=2):
#     pipe = model.get_pipe()
#     device = model.get_device()
#     tokenizer = pipe.tokenizer_2
#     float_type = pipe.dtype
#
#     batch_size = len(prompts)
#
#     token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
#     anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(n_anchors)))
#
#     default_attention_store_kwargs = {'token_indices': token_indices, 'mask_dropout': mask_dropout, 'extended_mapping': anchor_mappings}
#
#
#     return None, None



# def run_batch(seed=40, mask_dropout=0.5, style="A photo of ", subject="a cute dog", concept_token=['dog'], settings=["sitting in the beach", "standing in the snow"], out_dir=None):
#     flux_model = FLUXModel("black-forest-labs/FLUX.1-dev")
#     prompts = [f'{style}{subject} {setting}' for setting in settings]
#     images, image_all = run_batch_generation(flux_model, prompts, concept_token, seed, mask_dropout=mask_dropout)
#     return None, None



# def show_heatmap(pipe, image, prompt):
#     agg_attn = pipe.attention_store.aggregate_attention().float().cpu()
#     tokens_text = [pipe.tokenizer_2.decode(x) for x in pipe.tokenizer_2(prompt, padding="max_length", return_tensors="pt").input_ids[0]]
#     tokens_text = [f"{x}_{i}" for i, x in enumerate(tokens_text)]
#     idx_range = (0, 20)
#     visualize_tokens_attentions(agg_attn[0, idx_range[0]:idx_range[1]], tokens_text[idx_range[0]:idx_range[1]], image, prompt, heatmap_interpolation="bilinear")


# def show_distribution(pipe, prompt):
#     agg_attn = pipe.attention_store.aggregate_attention().float().cpu()
#     print(agg_attn.shape)
#     sum_attn_per_token = agg_attn.view(PROMPT_LEN, -1).mean(dim=1).cpu()
#     sum_attn_per_token = sum_attn_per_token / sum_attn_per_token.sum()
#     tokens_text = [pipe.tokenizer_2.decode(x) for x in pipe.tokenizer_2(prompt, padding="max_length", return_tensors="pt").input_ids[0]][:PROMPT_LEN]
#
#     # Show a bar plot of the attention per token
#     attn_per_token = {f'{t}_{i}': sum_attn_per_token[i] for i, t in enumerate(tokens_text)}
#
#     plt.figure(figsize=(100, 30))
#     plt.bar(attn_per_token.keys(), attn_per_token.values())
#     plt.xticks(rotation=90)
#     output_path = prompt + "_distribution.png"
#     plt.savefig(output_path, bbox_inches="tight")
#     print(f"Plot saved to {output_path}")
#     # plt.show()





# def create_token_indices(prompts, batch_size, concept_token, tokenizer):
#     if isinstance(concept_token, str):
#         concept_token = [concept_token]
#
#     concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
#     tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']
#
#     token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
#     for i, token_id in enumerate(concept_token_id):
#         batch_loc, token_loc = torch.where(tokens == token_id)
#         token_indices[i, batch_loc] = token_loc
#
#     return token_indices
#
# def create_anchor_mapping(bsz, anchor_indices=[0]):
#     anchor_mapping = torch.eye(bsz, dtype=torch.bool)
#     for anchor_idx in anchor_indices:
#         anchor_mapping[:, anchor_idx] = True
#
#     return anchor_mapping


import pandas as pd

def count_sentences(story):
    return len(story) if isinstance(story, list) else 0


def read_prompts_from_stories(stories_path):
    df = pd.read_csv(stories_path)
    df["full_stories"] = df["full_stories"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df = df.drop_duplicates(subset=["story_id"], keep="first")
    # df["num_sentences"] = df["full_stories"].apply(count_sentences)
    # filtered_df = df[df["num_sentences"].isin([5])]
    return df["full_stories"].tolist()

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

    prompts = read_prompts_from_stories('stories/full_stories.csv')
    # prompts = read_prompts()[:5]
    # prompts = [prompt for _, _, prompt in prompts]
    test_model(prompts)
