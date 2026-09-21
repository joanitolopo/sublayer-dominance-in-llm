from dataclasses import dataclass, field
import numpy as np
from typing import List


@dataclass
class WordActivation:
  res_in : np.ndarray
  attn: np.ndarray
  mlp: np.ndarray
  res_out: np.ndarray

@dataclass
class LanguageActivations:
  text: str
  words: List[str]
  layer_data: List[List[WordActivation]] = field(default_factory=list)
  num_layers: int = 0

  def get(self, layer_idx: int, word_idx: int) -> WordActivation:
    return self.layer_data[layer_idx][word_idx]

@dataclass
class AlignedRecord:
  pivot_word: str
  target_word: str
  pivot_idx: int
  target_idx: int
  pivot_layers: List[WordActivation]
  target_layers: List[WordActivation]

@dataclass
class LanguagePairResult:
  pivot_lang: str
  target_lang: str
  pivot_text: str
  target_text: str
  aligned_records: List[AlignedRecord]
  unaligned_pivot_indices: List[int]
  
@dataclass
class MergedPairResult:
  target_lang: str
  aligned_records: List[AlignedRecord] = field(default_factory=list)
