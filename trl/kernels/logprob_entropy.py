# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


_BLOCK_SIZE = 1024


@triton.jit
def _forward_kernel(
    logits_ptr,
    index_ptr,
    row_mask_ptr,
    logprobs_ptr,
    entropy_ptr,
    row_max_ptr,
    log_denom_ptr,
    mu_ptr,
    logits_batch_stride,
    logits_row_stride,
    index_batch_stride,
    index_row_stride,
    mask_batch_stride,
    mask_row_stride,
    n_cols: tl.constexpr,
    rows_per_batch,
    inv_t,
    HAS_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    row_in_batch = row - batch * rows_per_batch
    if HAS_MASK:
        is_valid = tl.load(row_mask_ptr + batch * mask_batch_stride + row_in_batch * mask_row_stride)
        if is_valid == 0:
            tl.store(logprobs_ptr + row, 0.0)
            tl.store(entropy_ptr + row, 0.0)
            tl.store(row_max_ptr + row, 0.0)
            tl.store(log_denom_ptr + row, 0.0)
            tl.store(mu_ptr + row, 0.0)
            return
    logits_row_ptr = logits_ptr + batch * logits_batch_stride + row_in_batch * logits_row_stride
    index = tl.load(index_ptr + batch * index_batch_stride + row_in_batch * index_row_stride)

    row_max = -float("inf")
    s = 0.0
    w = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_row_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        grew = new_max > row_max
        first = row_max == -float("inf")
        d = tl.where(first, 0.0, (new_max - row_max) * inv_t)
        c = tl.where(first, 0.0, tl.exp(-d))
        q = tl.where(mask, (new_max - x) * inv_t, 0.0)
        e = tl.where(mask, tl.exp(-q), 0.0)
        below = mask & (x < new_max) & (x > -float("inf"))
        ties = tl.sum((mask & (x == new_max)).to(tl.float32), axis=0)
        lower = tl.sum(tl.where(below, e, 0.0), axis=0)
        block_w = tl.sum(tl.where(below, q * e, 0.0), axis=0)
        old_total = 1.0 + s
        s = tl.where(grew, old_total * c + lower + (ties - 1.0), s + lower + ties)
        w = tl.where(grew, c * (w + d * old_total), w) + block_w
        row_max = new_max

    log_denom = libdevice.log1p(s)
    mu = w / (1.0 + s)
    x_t = tl.load(logits_row_ptr + index).to(tl.float32)
    q_t = (row_max - x_t) * inv_t
    tl.store(logprobs_ptr + row, -(q_t + log_denom))
    tl.store(entropy_ptr + row, log_denom + mu)
    tl.store(row_max_ptr + row, row_max)
    tl.store(log_denom_ptr + row, log_denom)
    tl.store(mu_ptr + row, mu)


@triton.jit
def _backward_kernel(
    logits_ptr,
    index_ptr,
    row_mask_ptr,
    row_max_ptr,
    log_denom_ptr,
    mu_ptr,
    grad_logprobs_ptr,
    grad_entropy_ptr,
    grad_logits_ptr,
    logits_batch_stride,
    logits_row_stride,
    index_batch_stride,
    index_row_stride,
    mask_batch_stride,
    mask_row_stride,
    grad_logits_batch_stride,
    grad_logits_row_stride,
    n_cols: tl.constexpr,
    rows_per_batch,
    inv_t,
    HAS_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LOGPROB_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
):
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    batch = row // rows_per_batch
    row_in_batch = row - batch * rows_per_batch
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    logits_row_ptr = logits_ptr + batch * logits_batch_stride + row_in_batch * logits_row_stride
    grad_logits_row_ptr = grad_logits_ptr + batch * grad_logits_batch_stride + row_in_batch * grad_logits_row_stride
    if HAS_MASK:
        is_valid = tl.load(row_mask_ptr + batch * mask_batch_stride + row_in_batch * mask_row_stride)
        if is_valid == 0:
            tl.store(grad_logits_row_ptr + offsets, 0.0, mask=mask)
            return
    x = tl.load(logits_row_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    index = tl.load(index_ptr + batch * index_batch_stride + row_in_batch * index_row_stride)
    row_max = tl.load(row_max_ptr + row)
    log_denom = tl.load(log_denom_ptr + row)
    q = tl.where(mask, (row_max - x) * inv_t, 0.0)
    log_p = -(q + log_denom)
    probabilities = tl.where(mask, tl.exp(log_p), 0.0)

    grad = tl.zeros((BLOCK_SIZE,), tl.float32)
    if HAS_LOGPROB_GRAD:
        grad_logprob = tl.load(grad_logprobs_ptr + row).to(tl.float32)
        x_t = tl.load(logits_row_ptr + index).to(tl.float32)
        target_grad = -libdevice.expm1(-((row_max - x_t) * inv_t + log_denom))
        grad += grad_logprob * tl.where(offsets == index, target_grad, -probabilities)
    if HAS_ENTROPY_GRAD:
        grad_entropy = tl.load(grad_entropy_ptr + row).to(tl.float32)
        mu = tl.load(mu_ptr + row)
        grad += grad_entropy * tl.where(probabilities > 0.0, probabilities * (q - mu), 0.0)
    tl.store(grad_logits_row_ptr + offsets, grad * inv_t, mask=mask)


def _layout(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return 1, 1, 0, 0
    if tensor.ndim == 2:
        return 1, tensor.shape[0], 0, tensor.stride(0)
    return tensor.shape[0], tensor.shape[1], tensor.stride(0), tensor.stride(1)


def _row_strides(tensor: torch.Tensor) -> tuple[int, int]:
    # Same as `_layout(tensor.unsqueeze(-1))[2:]`, but torch 2.6's Dynamo rejects `stride()` on a tensor created in
    # an autograd.Function backward.
    if tensor.ndim == 0:
        return 0, 0
    if tensor.ndim == 1:
        return 0, tensor.stride(0)
    return tensor.stride(0), tensor.stride(1)


class _LogProbEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        index: torch.Tensor,
        temperature: float,
        row_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, rows_per_batch, logits_batch_stride, logits_row_stride = _layout(logits)
        index_batch_stride, index_row_stride = _row_strides(index)
        if row_mask is None:
            row_mask_ptr = index
            mask_batch_stride = mask_row_stride = 0
        else:
            row_mask_ptr = row_mask
            mask_batch_stride, mask_row_stride = _row_strides(row_mask)
        n_rows = batch_size * rows_per_batch
        n_cols = logits.shape[-1]
        logprobs = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
        entropy = torch.empty_like(logprobs)
        row_max = torch.empty_like(logprobs)
        log_denom = torch.empty_like(logprobs)
        mu = torch.empty_like(logprobs)

        _forward_kernel[(n_rows,)](
            logits,
            index,
            row_mask_ptr,
            logprobs,
            entropy,
            row_max,
            log_denom,
            mu,
            logits_batch_stride,
            logits_row_stride,
            index_batch_stride,
            index_row_stride,
            mask_batch_stride,
            mask_row_stride,
            n_cols=n_cols,
            rows_per_batch=rows_per_batch,
            inv_t=1.0 / temperature,
            HAS_MASK=row_mask is not None,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(logits, index, row_max, log_denom, mu, row_mask)
        ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None):
        logits, index, row_max, log_denom, mu, row_mask = ctx.saved_tensors
        # Read the strides off the unpacked tensors, not forward's: saved-tensor hooks such as activation offloading
        # can return a sliced view as a contiguous copy.
        batch_size, rows_per_batch, logits_batch_stride, logits_row_stride = _layout(logits)
        n_rows = batch_size * rows_per_batch
        n_cols = logits.shape[-1]
        index_batch_stride, index_row_stride = _row_strides(index)
        if row_mask is None:
            row_mask_ptr = index
            mask_batch_stride = mask_row_stride = 0
        else:
            row_mask_ptr = row_mask
            mask_batch_stride, mask_row_stride = _row_strides(row_mask)
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        grad_logits_batch_stride, grad_logits_row_stride = rows_per_batch * n_cols, n_cols

        has_logprob_grad = grad_logprobs is not None
        has_entropy_grad = grad_entropy is not None
        if grad_logprobs is None:
            grad_logprobs = row_max
        else:
            grad_logprobs = grad_logprobs.contiguous()
        if grad_entropy is None:
            grad_entropy = mu
        else:
            grad_entropy = grad_entropy.contiguous()
        _backward_kernel[(n_rows, triton.cdiv(n_cols, _BLOCK_SIZE))](
            logits,
            index,
            row_mask_ptr,
            row_max,
            log_denom,
            mu,
            grad_logprobs,
            grad_entropy,
            grad_logits,
            logits_batch_stride,
            logits_row_stride,
            index_batch_stride,
            index_row_stride,
            mask_batch_stride,
            mask_row_stride,
            grad_logits_batch_stride,
            grad_logits_row_stride,
            n_cols=n_cols,
            rows_per_batch=rows_per_batch,
            inv_t=1.0 / ctx.temperature,
            HAS_MASK=row_mask is not None,
            BLOCK_SIZE=_BLOCK_SIZE,
            HAS_LOGPROB_GRAD=has_logprob_grad,
            HAS_ENTROPY_GRAD=has_entropy_grad,
        )
        return grad_logits, None, None, None


def selective_log_softmax_and_entropy(
    logits: torch.Tensor,
    index: torch.Tensor,
    temperature: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selected log-probabilities and Shannon entropy in one pass over logits."""
    return _LogProbEntropyFunction.apply(logits, index, temperature, row_mask)
