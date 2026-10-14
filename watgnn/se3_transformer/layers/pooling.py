# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import Dict, Literal

import torch.nn as nn
from torch import Tensor

from se3_transformer.graph import Graph


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    If you want max-pooling for type > 0 features, look into Vector Neurons.
    """

    def __init__(self, feat_type: int = 0, pool: Literal["max", "avg"] = "max"):
        """
        :param feat_type: Feature type to pool
        :param pool: Type of pooling: max or avg
        """
        super().__init__()
        assert pool in ["max", "avg"], f"Unknown pooling: {pool}"
        assert (
            feat_type == 0 or pool == "avg"
        ), "Max pooling on type > 0 features will break equivariance"
        self.feat_type = feat_type
        self.pool = pool

    def forward(self, features: Dict[str, Tensor], graph: Graph, **kwargs) -> Tensor:
        # A Graph is one graph, so the readout is over all of its nodes.
        feat = features[str(self.feat_type)]
        if self.pool == "max":
            pooled = feat.max(dim=0, keepdim=True).values
        else:
            pooled = feat.mean(dim=0, keepdim=True)
        return pooled.squeeze(dim=-1)
