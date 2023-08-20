import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Optional, List

class LoRALayer():
    def __int__(
        self,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p = lora_alpha)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: float = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=0, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight (embedding) matrix
            # Bias is still trainable due to additional task performance shown in Bitfit paper and minimal overhead.
            self.weight.requires_grad = False

        # initialize embedding and LoRA metrix weights correctly.
        self.reset_parameters()

    # overrides method of nn.Embedding because we need to initialize lora_A and lora_B weights also.
    def reset_parameters(self) -> None:
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as default for nn.Linear layer and B with zero metrix.
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)


    def forward(self, x: Tensor) -> Tensor:
        # shape of x: [B, T], then result: [B, T, d]
        result = nn.Embedding.forward(self, x)

        # if weights are not already merged, then add the lora component. else just return result
        if self.r > 0 and not self.merged:
            # shape = [B, T, r]
            # F.embedding() is embedding look up api
            after_A = F.embedding(x, self.lora_A.transpose(0, 1),
                        self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            # shape: [B, T, d]
            after_AxB = (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            # shape: [B, T, d]
            result += after_AxB

        return result

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            # Training mode: unmerge the weights and mark it.
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).traspose(0, 1) * self.scaling
            self.merged = False
        else:
            # Eval mode: merge the weights and mark it.
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).traspose(0, 1) * self.scaling
            self.merged = True


class Linear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: float = 1,
            lora_dropout: float = 0,
            fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r

            # Freezing the linear layer weight matrix.
            # Bias is still trainable due to additional task performance shown in Bitfit paper and minimal overhead.
            self.weight.requires_grad = False

        # initialize original linear layer and LoRA metrix weights correctly.
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:

        def T(w):
            return w.transpose(0, 1 ) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), self.bias)
        # Alternate way, if no change to x is needed: nn.Linear.forward(self, x)

        # if weights are not already merged, then add the lora component.
        if self.r > 0 and not self.merged:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # set model in linear layer
        nn.Linear.train(self, mode)
        if mode:
            # Training mode: unmerge the weights and mark it.
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # Eval/Inference mode: merge the weights and mark it.
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True


class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        assert out_features % len(enable_lora) == 0, 'Length of {enable_lora} should be divisible by {out_features}'

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r = r, lora_alpha = lora_alpha,
            lora_dropout = lora_dropout, merge_weights = merge_weights
            )

        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            # each Q and V of shape: [r, in_features]
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            # Weights for conv1d operation with #groups = sum(enable_lora)
            # each Q and V of shape: [out_features, r]
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the linear layer weight matrix.
            # Bias is still trainable due to additional task performance shown in Bitfit paper and minimal overhead.
            self.weight.requires_grad = False

            # Assign indices at which lora is applied.
            self.lora_indices = self.assign_lora_indices()

        # Weights initialization
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def assign_lora_indices(self):
        # Track indices at which lora module is applied.
        # In case of applying it to transformer MHA, shape: [len(enable_lora), emb_dim]
        lora_indices = self.weight.new_zeros(
            (self.out_features,), dtype = torch.bool
        ).view(len(self.enable_lora), -1)
        lora_indices[self.enable_lora, :] = True
        # shape: (out_features,)
        lora_indices = self.lora_indices.view(-1)
        return lora_indices

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        # [out_features, in_features] with all zeros
        result = x.new_zeros((len(self.lora_indices), *x[1:]))
        # fill values at places where lora is applied else keep it zero.
        result[self.lora_indices] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # apply 1d convolution with kernel size = 1 (essentially matrix multiplication of A & B) separately for Q, K, and V.
        # delta_w shape: [out_features//len(enable_lora) * sum(enable_lora), in_features]
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),     # [1, r * sum(enable_lora), in_feature]
            self.lora_B.unsqueeze(-1),    # [out_features//len(enable_lora) * sum(enable_lora), r, 1]
            groups = sum(self.enable_lora)
        ).squeeze(0)

        # [in_features, out_features] if self.fan_in_fan_out is True else [out_features, in_features]
        return T(self.zero_pad(delta_w))

    def forward(self, x: Tensor) -> Tensor:

        def T(w):
            return w.transpose(0, 1 ) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), self.bias)
        # Alternate way, if no change to x is needed: nn.Linear.forward(self, x)

        # if weights are not already merged, then add the lora component.
        if self.r > 0 and not self.merged:
            result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
        return result

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # set model in linear layer
        nn.Linear.train(self, mode)
        if mode:
            # Training mode: unmerge the weights and mark it.
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            # Eval/Inference mode: merge the weights and mark it.
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

