from simalign import SentenceAligner
from typing import Dict, List

from schema import LanguageActivations, LanguagePairResult, AlignedRecord, MergedPairResult
from hooks import ActivationExtractor

class WordAligner:
  def __init__(self, model: str = "xlmr", token_type: str = "bpe", matching_methods: str = "mai"):
    self._matching_key = {"mai": "itermax", "mwmf": "mwmf", "inter": "inter"}.get(
        matching_methods, "itermax"
    )
    self.aligner = SentenceAligner(
        model=model, token_type=token_type, matching_methods=matching_methods
    )
  def align(self, pivot_text: str, target_text: str) -> Dict:
    pivot_words = pivot_text.split()
    target_words = target_text.split()
    raw = self.aligner.get_word_aligns(pivot_words, target_words)

    pairs = raw.get(self._matching_key) or next(iter(raw.values()))

    aligned_pivot_indices = {p for p, _ in pairs}
    unaligned = [i for i in range(len(pivot_words)) if i not in aligned_pivot_indices]

    return {
        "pivot_words": pivot_words,
        "target_words": target_words,
        "pairs": pairs,
        "unaligned_pivot_indices": unaligned,
    }

class AlignedActivationCollector:
  def __init__(self, extractor: ActivationExtractor, aligner: WordAligner):
    self.extractor = extractor
    self.aligner = aligner
    self._activation_cache: Dict[str, LanguageActivations] = {}

  def _get_activations(self, lang: str, text: str) -> LanguageActivations:
    cache_key = f"{lang}::{text[:50]}"
    if cache_key not in self._activation_cache:
      self._activation_cache[cache_key] = self.extractor.extract(text)
    return self._activation_cache[cache_key]

  def collect_pair(self, pivot_lang: str, target_lang: str, pivot_text: str,
                   target_text: str) -> LanguagePairResult:
    pivot_act = self._get_activations(pivot_lang, pivot_text)
    target_act = self._get_activations(target_lang, target_text)

    alignment = self.aligner.align(pivot_text, target_text)

    records: List[AlignedRecord] = []
    for p_idx, t_idx in alignment["pairs"]:
      if p_idx >= len(pivot_act.words) or t_idx >= len(target_act.words):
          continue
      records.append(
          AlignedRecord(
              pivot_word=pivot_act.words[p_idx],
              target_word=target_act.words[t_idx],
              pivot_idx=p_idx,
              target_idx=t_idx,
              pivot_layers=[layer[p_idx] for layer in pivot_act.layer_data],
              target_layers=[layer[t_idx] for layer in target_act.layer_data],
          )
      )

    return LanguagePairResult(
        pivot_lang=pivot_lang,
        target_lang=target_lang,
        pivot_text=pivot_text,
        target_text=target_text,
        aligned_records=records,
        unaligned_pivot_indices=alignment["unaligned_pivot_indices"],
    )
  def collect_row( self, dataset, row_idx: int, pivot_lang: str,
                  target_langs: List[str], split: str = "train") -> Dict[str, LanguagePairResult]:
    pivot_text = dataset[split][pivot_lang][row_idx]

    results = {}
    for lang in target_langs:
      target_text = dataset[split][lang][row_idx]
      results[lang] = self.collect_pair(pivot_lang, lang, pivot_text, target_text)
    return results

def collect_multiple_rows(collector: AlignedActivationCollector,
                          dataset,
                          row_indices: List[int],
                          pivot_lang: str,
                          target_langs: List[str],
                          split: str = "train") -> Dict[str, MergedPairResult]:
  """
  Build P = union FLORES cross row aligned_records on `row_indices`, separated by
  language target L_i in target langs. This is a word pair set (w_p, w_i) which used
  to estimate hat_ell(L_i | L_p) in 4.2 section
  """
  merged: Dict[str, MergedPairResult] = {
      lang: MergedPairResult(target_lang=lang) for lang in target_langs
  }

  for row_idx in row_indices:
    row_results = collector.collect_row(dataset, row_idx=row_idx, pivot_lang=pivot_lang,
                                        target_langs=target_langs, split=split)
    for lang in target_langs:
      merged[lang].aligned_records.extend(row_results[lang].aligned_records)

  return merged