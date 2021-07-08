"""ssa_utils.

Helper file with some SSA utils to import
"""
import torch
import numpy as np
from typing import Callable

from torch import nn
from scipy.special import softmax

### Dist functions


def norm_dist_matrix(x1, x2, norm_fn):
    """Compute norm dist matrix"""
    return norm_fn(x1, x2)
    # return norm_fn(x1[:, None, :] - x2[None, :, :])


def l1_dist_matrix(x1, x2):
    """Compute l1 norm"""
    l1_norm = lambda x, y: np.sum(np.abs(x[:, None, :] - y[None, :, :]), -1)
    return norm_dist_matrix(x1, x2, l1_norm)


def l1_dist_matrix_torch(x1, x2):
    l1_norm = lambda x, y: torch.sum(torch.abs(x[:, None, :] - y[None, :, :]), -1)
    return norm_dist_matrix(x1, x2, l1_norm)


def get_alpha(norm_mat):
    return softmax(-norm_mat, axis=1)


def get_alpha_torch(norm_mat):
    return nn.functional.softmax(-norm_mat, dim=1)


def get_beta(norm_mat):
    return softmax(-norm_mat, axis=0)


def get_beta_torch(norm_mat):
    return nn.functional.softmax(-norm_mat, dim=0)


def get_ssa(alpha, beta):
    attn = alpha + beta - alpha * beta
    return attn


def get_dist(norm_mat, attn):
    # Calculate dist and normalize by alignment length
    dist_mat = np.einsum("ij,ij->ij", norm_mat, attn)
    dist = np.sum(dist_mat)

    align_length = np.sum(attn)
    norm_dist = dist / align_length
    return dist_mat, norm_dist


def get_dist_torch(norm_mat, attn):
    # Calculate dist and normalize by alignment length
    dist_mat = torch.einsum("ij,ij->ij", norm_mat, attn)
    dist = torch.sum(dist_mat)

    align_length = torch.sum(attn)
    norm_dist = dist / align_length
    return dist_mat, norm_dist


def soft_pool_to_ref(
    targ_embed: np.ndarray, ref_embed: np.ndarray, ref_positions: np.ndarray
) -> np.ndarray:
    """soft_pool_to_ref.

    Compute an attention score of the target embedding to each reference
    sequence positional embedding.
    This is used in the SSA_Test file

    Args:
        targ_embed (np.ndarray): A positional embedding of dim l x d for a
            target sequence
        ref_embed (np.ndarray): A positional embedding of dim l x d for a
            reference sequence embedding
        ref_positions (np.ndarray): Positions to pool over in the REFERENCE
            sequence of shape l'
    Return:
        np.ndarray: Pooled target of shape d
    """

    # Only select the impt residues of ref embd
    ref_embed = ref_embed[ref_positions]

    # Reference
    norm_ref_1 = l1_dist_matrix(targ_embed, ref_embed)
    alpha_1_ref = get_alpha(norm_ref_1)
    beta_1_ref = get_beta(norm_ref_1)
    a_1_ref = get_ssa(alpha_1_ref, beta_1_ref)

    # Attentively pool over length of input sequence
    # Output is num of ref positions by dimension
    finished_embed = np.einsum("ij,id->jd", beta_1_ref, targ_embed)

    # Avg pool over ref seq
    finished_embed = finished_embed.mean(0)
    return finished_embed


def get_ssa_dist(
    embedding_1: torch.tensor, embedding_2: torch.tensor, norm_fn: Callable
):
    """get_ssa_dist.

    Compute soft symmetric attention between two sequences.

    Embedding 1 and embedding 2 should already be on GPU if cuda is to be used.

    Args:
        embedding_1 (torch.tensor): First embedding (test obj)
        embedding_2 (torch.tensor): second embedding (ref obj)
        norm_fn (Callable): Callable function to call on the dist matrix
    Return:
        attn_matrix, distance matrix, noramlized distance

    """
    # Apply norm fn
    norm_mat = norm_dist_matrix(embedding_1, embedding_2, norm_fn)

    # Calculate row attn and col attn
    alpha = nn.functional.softmax(-norm_mat, dim=1)
    beta = nn.functional.softmax(-norm_mat, dim=0)

    # Calculate symmetric attn
    attn = alpha + beta - alpha * beta

    # Calculate align length
    align_length = torch.sum(attn)

    # Calculate dist and normalize by alignment length
    dist = torch.einsum("ij,ij->", norm_mat, attn)
    norm_dist = dist / align_length
    norm_dist = norm_dist.cpu().numpy().item()

    return attn, dist, norm_dist


def get_rssa_dist(
    embedding_1: torch.tensor,
    embedding_2: torch.tensor,
    ref_embedding: torch.tensor,
    norm_fn: Callable,
    and_gate: bool = False,
):
    """get_rssa_dist.

    Compute attentive pooling with respect to some third ref reference
    embedding.

    Embedding 1, embedding 2, and the ref embedding should already be on GPU if
    cuda is to be used. Additionally, ref embedding should already be subsetted
    to the desired ref positions to use this.

    Args:
        embedding_1 (torch.tensor): First embedding
        embedding_2 (torch.tensor): second embedding
        ref_embedding (torch.tensor): Reference embedding to compute attention.
            If this is to be subsetted to a smaller set of residues, this
            should already be done.
        norm_fn (Callable): Callable function to call on the dist matrix
        and_gate (bool): If true, also use an "and" gate by multiplying cross
            attention with the normal soft symmetric attention weights
    Return:
        cross attention, distance matrix, normalized distance

    """
    # Norm of distance between train and test sequence
    norm_mat = norm_dist_matrix(embedding_1, embedding_2, norm_fn)

    # Now compute beta to a third, reference sequence
    # Only compute beta because we only want to compute
    norm_ref_1 = l1_dist_matrix_torch(embedding_1, ref_embedding)
    beta_1_ref = get_beta_torch(norm_ref_1)

    norm_ref_2 = l1_dist_matrix_torch(embedding_2, ref_embedding)
    beta_2_ref = get_beta_torch(norm_ref_2)

    ## Use only beta for cross ref!
    # Now we have an an attention for looking at
    # residue specific distances
    # This uses reference attention to compute distance
    cross_attn = torch.einsum("il,jl->ij", beta_1_ref, beta_2_ref)

    # If we have an and gate
    if and_gate:

        # Calculate row attn and col attn
        alpha = nn.functional.softmax(-norm_mat, dim=1)
        beta = nn.functional.softmax(-norm_mat, dim=0)

        # Calculate symmetric attn
        attn = alpha + beta - alpha * beta
        cross_attn = cross_attn * attn

    # Now attentively pool and sum distances
    dist, norm_dist = get_dist_torch(norm_mat, cross_attn)

    norm_dist = norm_dist.cpu().numpy().item()

    return cross_attn, dist, norm_dist


if __name__ == "__main__":
    pass
