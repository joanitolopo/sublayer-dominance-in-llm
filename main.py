import os
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel

from get_language_vector import LanguageDirectionEstimator

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Error: HF_TOKEN is missing! Did you create your .env file?")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "CohereLabs/tiny-aya-base"
    dataset_id = "joanitolopo/flores-plus-indo_europa_germanic"
    pivot_lang = "English"

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModel.from_pretrained(model_id, token=hf_token).to(device)

    # instantiate class
    est = LanguageDirectionEstimator(model, tokenizer, device=device)


if __name__=="main":
    main()