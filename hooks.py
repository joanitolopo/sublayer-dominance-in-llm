import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

# import schema.py
from schema import LanguageActivations, WordActivation

class ActivationExtractor:
  def __init__(self, model, tokenizer, device: str = "cpu"):
    self.model = model
    self.tokenizer = tokenizer
    self.device = device
    self.num_layers = len(model.model.layers)
    self._raw_outputs: Dict[str, torch.Tensor] = {}
    self._hooks = []
    self._register_hooks()

  def _register_hooks(self):
    def make_hook(name):
      def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self._raw_outputs[name] = out.detach().cpu()
      return hook

    for layer_idx in range(self.num_layers):
      layer = self.model.model.layers[layer_idx]
      h1 = layer.self_attn.o_proj.register_forward_hook(make_hook(f'attn_out_layer_{layer_idx}'))
      h2 = layer.mlp.down_proj.register_forward_hook(make_hook(f'mlp_out_layer_{layer_idx}'))
      self._hooks.extend([h1, h2])

  def remove_hooks(self):
    for h in self._hooks:
      h.remove()
    self._hooks = []

  @staticmethod
  def _aggregate_subwords_to_words(token_matrix: np.ndarray, offset_mapping: List[Tuple[int, int]],
                                   text: str) -> Tuple[List[str], np.ndarray]:
    words = text.split()
    word_spans = []
    cursor = 0
    for word in words:
      start = text.find(word, cursor)
      end = start + len(word)
      word_spans.append((start, end))
      cursor = end

    word_vectors = []
    for w_start, w_end in word_spans:
      subword_idx = [
          i for i, (s, e) in enumerate(offset_mapping) if s != e and s >= w_start and e <= w_end]
      if subword_idx:
          vec = token_matrix[subword_idx].mean(axis=0)
      else:
          vec = np.zeros(token_matrix.shape[1])
      word_vectors.append(vec)

    return words, np.array(word_vectors)

  def extract(self, text:str) -> LanguageActivations:
    inputs = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    self._raw_outputs.clear()
    with torch.no_grad():
      outputs = self.model(**inputs, output_hidden_states=True)

    words_ref = []
    layer_data: List[List[WordActivation]] = []

    for i in range(self.num_layers):
      res_in_sub = outputs["hidden_states"][i].squeeze(0).float().cpu().numpy()
      res_out_sub = outputs["hidden_states"][i + 1].squeeze(0).float().cpu().numpy()
      attn_sub = self._raw_outputs[f"attn_out_layer_{i}"].squeeze(0).float().numpy()
      mlp_sub = self._raw_outputs[f"mlp_out_layer_{i}"].squeeze(0).float().numpy()

      words, res_in_w = self._aggregate_subwords_to_words(res_in_sub, offset_mapping, text)
      _, attn_w = self._aggregate_subwords_to_words(attn_sub, offset_mapping, text)
      _, mlp_w = self._aggregate_subwords_to_words(mlp_sub, offset_mapping, text)
      _, res_out_w = self._aggregate_subwords_to_words(res_out_sub, offset_mapping, text)

      words_ref.extend(words)
      layer_words = [
          WordActivation(
              res_in=res_in_w[wi],
              attn=attn_w[wi],
              mlp=mlp_w[wi],
              res_out=res_out_w[wi],
              )
                for wi in range(len(words))]
      layer_data.append(layer_words)

    return LanguageActivations(text=text, words=words_ref, layer_data=layer_data,
                               num_layers=self.num_layers)
