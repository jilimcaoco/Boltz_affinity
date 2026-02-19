"""
Boltz Affinity Rescoring Module.

Production-ready protein-ligand affinity rescoring system
leveraging the Boltz-2 affinity module without requiring
diffusion-based structure prediction.

Usage:
    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer("boltz2_aff.ckpt")
    result = rescorer.rescore_pdb("complex.pdb")
    print(f"Affinity: {result.affinity_pred:.2f} ± {result.affinity_std:.2f}")
"""

from boltz.affinity_rescoring.rescorer import AffinityRescorer

__all__ = ["AffinityRescorer"]
__version__ = "1.0.0"
