import torch
import torch_npu
import torch.distributed as dist


def all_to_all_4d(
    input_tensor: torch.tensor, group=None
) -> torch.tensor:
    seq_world_size = dist.get_world_size(group)

    # Transfer shape (bs, seqlen/sp, hc, hs) to (bs, seqlen, hc/sp, hs)
    bs, shard_seqlen, hc, hs = input_tensor.shape
    seqlen = shard_seqlen * seq_world_size
    shard_hc = hc // seq_world_size

    input_t = (
        input_tensor.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        .transpose(0, 2)
        .contiguous()
    )

    output = torch.empty_like(input_t)
    if seq_world_size > 1:
        dist.all_to_all_single(output, input_t, group=group)
    else:
        output = input_t
    
    output = output.reshape(seqlen, bs, shard_hc, hs).transpose(0, 1).contiguous()
    return output

def all_to_all_3d(
    input_tensor: torch.tensor, group=None
) -> torch.tensor:
    seq_world_size = dist.get_world_size(group)

    # Transfer shape (seqlen, hc/sp, hs) to (seqlen/sp, hc, hs)
    seqlen, shard_hc, hs = input_tensor.shape
    hc = shard_hc * seq_world_size
    shard_seqlen = seqlen // seq_world_size

    input_t = (
        input_tensor.reshape(seq_world_size, shard_seqlen, shard_hc, hs)
        .transpose(1, 2)
        .contiguous()
    )

    output = torch.empty_like(input_t)
    if seq_world_size > 1:
        dist.all_to_all_single(output, input_t, group=group)
    else:
        output = input_t
    
    output = output.reshape(hc, shard_seqlen, hs).transpose(0, 1).contiguous()
    return output

def all_gather_2d(input_tensor: torch.tensor, world_size: int, group=None) -> torch.tensor:
    s, d = input_tensor.shape
    input_gather = torch.zeros(
        world_size * s, d, dtype=input_tensor.dtype, device=input_tensor.device
    )
    dist.all_gather_into_tensor(input_gather, input_tensor, group=group)

    return input_tensor

