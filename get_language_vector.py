import numpy as np
from typing import List, Tuple
import torch

from alignment import WordAligner, AlignedActivationCollector, collect_multiple_rows
from hooks import ActivationExtractor

def build_language_vocab_set(dataset, lang_col: str, tokenizer, other_lang_cols: List[str],
                             split: str = "train", n_samples: int = 50) -> Tuple[set, set]:
  """
  V_{L_i}: words that not include in lang_col, as identically string, on other language in
  other_lang_cols (exclusivity filter). Return (token_ids, exclusive_words.)
  token_ids = union subword decomposition of exclusive_words, use for W_U.
  """
  def words_of(col: str, n: int) -> set:
    words = set()
    for text in dataset[split][col][:n]:
      words.update(text.split())
    return words

  target_words = words_of(lang_col, n_samples)
  other_words = set()
  for col in other_lang_cols:
    other_words |= words_of(col, n_samples)

  exclusive_words = target_words - other_words

  token_ids = set()
  for word in exclusive_words:
    ids = tokenizer.encode(word, add_special_tokens=False)
    token_ids.update(ids)

  return token_ids, exclusive_words

def compute_w_hat(model, token_ids: set) -> Tuple[np.ndarray, np.ndarray]:
  """
  bar u_{L_i} = mean_{v in V_{L_i}} W_U[v, :]
  hat w(L_i)  = bar u_{L_i} / ||bar u_{L_i}||

  W_u obtained from model.lm_head.weight. Return (u_bar, w_hat)
  """
  W_U = model.lm_head.weight.detach()
  idx = torch.tensor(sorted(token_ids), dtype=torch.long)
  rows = W_U[idx].float()
  u_bar = rows.mean(dim=0)
  w_hat = u_bar / u_bar.norm()
  return u_bar.cpu().numpy(), w_hat.cpu().numpy()

class LanguageDirectionEstimator:
  """
  This wrapper for Residual-Space Attribution (RSA) and Vocaulary-Space Attribution (VSA).
  Both of them will produce hat direction in the same residual space (4.1. Dominance Estimation Procedure),
  but different estimation and independent.

  """

  def __init__(self, model, tokenizer, device: str = "cpu"):
    self.model = model
    self.tokenizer = tokenizer
    self.device = device
    self.num_layers = len(model.model.layers)

    # k* = K-1 (K-2 on hidden states 0..K), we avoid x_K
    # which already contain final layer normalization (4.4. Residual-Space Attribution)
    self.k_star = self.num_layers - 2

    # usefull for RSA and dominance calculation procedure
    self._extractor = None
    self._aligner = None
    self._collector = None

    self.rsa_results_: dict = {}
    self.vsa_results_: dict = {}

  # activation infra
  @property
  def extractor(self):
    if self._extractor is None:
      self._extractor = ActivationExtractor(self.model, self.tokenizer, self.device)
    return self._extractor

  @property
  def aligner(self):
    if self._aligner is None:
      self._aligner = WordAligner(model="xlmr", token_type="bpe", matching_methods="mai")
    return self._aligner

  @property
  def collector(self):
    if self._collector is None:
      self._collector = AlignedActivationCollector(self.extractor, self.aligner)
    return self._collector

  def close(self):
    """Revoke hooks if the instance is no longer in use"""
    if self._extractor is not None:
      self._extractor.remove_hooks()

  # RSA methods
  def rsa(self, dataset, pivot_lang: str, target_language: list, row_indices=None,
          split: str = "train", k_star: int | None = None, use: str = "res_out"):
    """
    bardelta_{k*} = (1/|P|) * sum_{(w_p,w_i) in P} (x_{k*}(w_i) - x_{k*}(w_p))
    hatell(L_i | L_p) = bardelta_{k*} / ||bardelta_{k*}||

    P = translation-aligned words pairs (SimAlign) from `dataset[split]`
    on `row_indices`. `k_star` default is self.k_star (K-1, automatically from the model num_layers)
    """
    if row_indices is None:
      row_indices = list(range(dataset.num_rows[split]))

    k_idx = self.k_star if k_star is None else k_star

    merged = collect_multiple_rows(self.collector, dataset, row_indices, pivot_lang=pivot_lang,
                                   target_langs = target_language, split=split)

    out = {}
    for lang in target_language:
      result = merged[lang]
      if len(result.aligned_records) == 0:
        out[lang] = None
        continue
      diffs = np.array([getattr(rec.target_layers[k_idx], use) - getattr(rec.pivot_layers[k_idx], use) for rec in result.aligned_records])
      delta_bar = diffs.mean(axis=0)
      ell_hat = delta_bar / np.linalg.norm(delta_bar)
      out[lang] = {
          "delta_bar": delta_bar,
          "ell_hat": ell_hat,
          "k_star": k_idx,
          "pivot_lang": pivot_lang,
          "n_records": len(result.aligned_records),
          "merged_result": result,}

    self.rsa_results_.update(out)
    return out

  def vsa(self, dataset, target_langs: list, split: str = "devtest", n_samples: int = 50):
    """
    V_{L_i}     = exclusive words L_i vs other lang in target_langs
    bar u_{L_i} = mean_{v in V_{L_i}} W_U[v, :]
    hat w(L_i)  = bar u_{L_i} / ||bar u_{L_i}||
    """
    out = {}
    for lang in target_langs:
      others = [l for l in target_langs if l != lang]
      token_ids, V_Li = build_language_vocab_set(dataset, lang_col=lang, tokenizer=self.tokenizer,
                                                 other_lang_cols=others, split=split, n_samples=n_samples,)
      u_bar, w_hat = compute_w_hat(self.model, token_ids)
      out[lang] = {
          "u_bar": u_bar,
          "w_hat": w_hat,
          "V_Li": V_Li,
          "token_ids": token_ids,
          "n_exclusive_words": len(V_Li),
      }
      print(f"{lang}: {len(V_Li)} eksklusif words -> {len(token_ids)} token id")

    self.vsa_results_.update(out)
    return out

  def get_vectors(self, method: str = "rsa", key: str | None = None):
    """
    Take {lang: vector} + {lang: n} from saved .rsa() or .vsa()
    """
    store = self.rsa_results_ if method == "rsa" else self.vsa_results_
    key = key or ("ell_hat" if method == "rsa" else "w_hat")
    vectors = {lang: (res[key] if res is not None else None) for lang, res in store.items()}
    n = {lang: (0 if res is None else res.get("n_records", res.get("n_exclusive_words", 0))) for lang, res in store.items()}
    return vectors, n