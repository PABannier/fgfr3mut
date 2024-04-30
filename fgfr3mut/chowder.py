# Copyright (c) Owkin, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import Optional, Union
import warnings


class Chowder(torch.nn.Module):
    """
    Chowder module.
    See https://arxiv.org/abs/1802.02212.

    Example:
        >>> module = Chowder(in_features=2048, out_features=1, n_extreme=100)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Input dimension of the extracted features.
    n_extreme: int
        Number of extreme tiles to aggregate.
    """

    def __init__(
        self,
        in_features: int,
        n_extreme: int,
    ):
        super(Chowder, self).__init__()

        self.score_model = TilesMLP(in_features, out_features=1)
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_extreme, n_bottom=n_extreme)

        self.mlp = MLP(2 * n_extreme, 1)

        self.mlp.apply(self.weight_initialization)

    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, 2*N_EXTREME, OUT_FEATURES)

        # Apply MLP to the 2*N_EXTREME scores
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        return y.squeeze(2), extreme_scores


class MLP(torch.nn.Sequential):
    """
    MLP Module

    Parameters
    ----------
    in_features: int
    out_features: int
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        d_model = in_features
        layers = []

        activation = torch.nn.Sigmoid()
        for h in [128, 64]:
            seq = [torch.nn.Linear(d_model, h, bias=True)]
            d_model = h

            if activation is not None:
                seq.append(activation)

            layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.

    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0


    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, mask_value={}, bias={}".format(
            self.in_features, self.out_features, self.mask_value, self.bias is not None
        )


class TilesMLP(torch.nn.Module):
    """
    MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
    ):

        super(TilesMLP, self).__init__()

        activation = torch.nn.Sigmoid()

        self.hidden_layers = torch.nn.ModuleList()
        for h in [128]:
            self.hidden_layers.append(
                MaskedLinear(in_features, h, mask_value="-inf")
            )
            self.hidden_layers.append(activation)
            in_features = h

        self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=True))

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x


class ExtremeLayer(torch.nn.Module):
    """
    Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles

    .. warning::
        If top tiles or bottom tiles is superior to the true number of tiles in the input then padded tiles will
        be selected and their value will be 0.

    Parameters
    ----------
    n_top: int
        number of top tiles to select
    n_bottom: int
        number of bottom tiles to select
    dim: int
        dimension to select top/bottom tiles from
    return_indices: bool
        Whether to return the indices of the extreme tiles
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not ((n_top is not None and n_top > 0) or (n_bottom is not None and n_bottom > 0)):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, ...)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, ...)

        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in the input then padded tiles will
        be selected and their value will be 0.

        Returns
        -------
        extreme_tiles: torch.Tensor
            (B, N_TOP + N_BOTTOM, ...)
        """

        if self.n_top and self.n_bottom and ((self.n_top + self.n_bottom) > x.shape[self.dim]):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                + f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self):
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"
