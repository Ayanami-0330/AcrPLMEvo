from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_model_id: str
    family: str
    seq_mode: str
    target_module_candidates: Tuple[str, ...]
    max_length: int
    batch_size: int


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "protbert": BackboneSpec(
        name="protbert",
        hf_model_id="Rostlab/prot_bert_bfd",
        family="bert",
        seq_mode="spaced",
        target_module_candidates=("query", "value"),
        max_length=1024,
        batch_size=16,
    ),
    "prott5": BackboneSpec(
        name="prott5",
        hf_model_id="Rostlab/prot_t5_xl_uniref50",
        family="t5",
        seq_mode="spaced",
        target_module_candidates=("q", "v"),
        max_length=1024,
        batch_size=2,
    ),
    "esm2": BackboneSpec(
        name="esm2",
        hf_model_id="facebook/esm2_t36_3B_UR50D",
        family="esm",
        seq_mode="plain",
        target_module_candidates=("q_proj", "v_proj", "query", "value"),
        max_length=1024,
        batch_size=2,
    ),
    "ankh": BackboneSpec(
        name="ankh",
        hf_model_id="ElnaggarLab/ankh-large",
        family="t5",
        seq_mode="spaced",
        target_module_candidates=("q", "v"),
        max_length=1024,
        batch_size=2,
    ),
}
