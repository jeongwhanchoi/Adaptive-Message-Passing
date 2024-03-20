from collections import OrderedDict
from typing import Callable, Tuple, Optional

import torch
from torch import Tensor, tanh
from torch.nn import Sequential, Linear, LeakyReLU, Module, ModuleList, Dropout, GELU
from torch_geometric.nn import GINConv, GCNConv, GINEConv, ResGatedGraphConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import OptPairTensor
from torch_sparse import SparseTensor

from adgn import AntiSymmetricConv


class EdgeFilterGINConv(GINConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size=None,
        activation=torch.nn.functional.tanh,
    ) -> torch.Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, edge_filter=edge_filter, size=None
        )

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return activation(self.nn(out))

    # def message(
    #     self,
    #     x_j: torch.Tensor,
    #     edge_weight: Optional[torch.Tensor] = None,
    #     edge_filter: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #
    #     # backward compatibility
    #     if edge_filter is not None and len(edge_filter.shape) == 1:
    #         edge_filter = edge_filter.view(-1, 1)
    #
    #     if edge_filter is not None:
    #         return edge_filter * x_j
    #     else:
    #         return x_j

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j


class EdgeFilterGINEConv(GINEConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size=None,
        activation=torch.nn.functional.tanh,
    ) -> torch.Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, edge_filter=edge_filter, size=None
        )

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return activation(self.nn(out))

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.lin is not None:
            edge_weight = self.lin(edge_weight.float())

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight + x_j
        else:
            return edge_filter * (edge_weight + x_j)


class EdgeFilterGatedGCNConv(ResGatedGraphConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size=None,
        activation=torch.nn.functional.tanh,
    ) -> torch.Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        k = self.lin_key(x[1])
        q = self.lin_query(x[0])
        v = self.lin_value(x[0])

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor)
        out = self.propagate(
            edge_index, k=k, q=q, v=v, edge_weight=None, edge_filter=edge_filter
        )

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return activation(out)

    # def message(
    #     self,
    #     k_i: Tensor,
    #     q_j: Tensor,
    #     v_j: Tensor,
    #     edge_weight: Optional[torch.Tensor] = None,
    #     edge_filter: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #
    #     x_j = self.act(k_i + q_j) * v_j  # the node message
    #
    #     # backward compatibility
    #     if edge_filter is not None and len(edge_filter.shape) == 1:
    #         edge_filter = edge_filter.view(-1, 1)
    #
    #     if edge_filter is not None:
    #         return edge_filter * x_j
    #     else:
    #         return x_j

    def message(
        self,
        k_i: Tensor,
        q_j: Tensor,
        v_j: Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_j = self.act(k_i + q_j) * v_j  # the node message

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j
        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j


class EdgeFilterGCNConv(GCNConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        activation=torch.nn.functional.tanh,
    ) -> torch.Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, edge_filter=edge_filter, size=None
        )

        if self.bias is not None:
            out = out + self.bias

        return activation(out)

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j


class LayerGenerator:
    def __init__(self, **kwargs):
        super(LayerGenerator, self).__init__()

    def make_generators(
        self, node_size, edge_size, hidden_size, output_size, **kwargs
    ) -> Tuple[Callable, Callable]:
        """
        Creates two hidden and output layer generators using the
        provided parameters. They both accept a layer id. If the layer id is 0,
        it is assumed that we are generating the layers for the input
        """
        raise NotImplementedError("To be implemented in a sub-class")


class RelationalConv(Module):
    """
    Wrapper that implements multiple convolutions at a given layer,
    one for each DISCRETE edge type. Breaks if continuous values are used
    """

    def __init__(self, edge_size, conv_layer, **kwargs):
        super(RelationalConv, self).__init__()
        self.edge_size = edge_size
        self.edge_convs = ModuleList()

        for _ in range(edge_size):
            self.edge_convs.append(conv_layer(**kwargs))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = 0

        for e, conv in enumerate(self.edge_convs):
            if edge_filter is None:
                outputs += tanh(conv(x, edge_index[:, edge_attr == e]))
            else:
                outputs += tanh(
                    conv(
                        x,
                        edge_index[:, edge_attr == e],
                        edge_filter=edge_filter[edge_attr == e],
                    )
                )

        return outputs


class ADGNGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Linear(node_size, hidden_size)
            else:
                if edge_size == 0:
                    # stacking Antisymmetric DGN convolutions rather than weight
                    # sharing (see Antisymmetric DGN paper)
                    return AntiSymmetricConv(
                        in_channels=hidden_size,
                        num_iters=1,
                        epsilon=kwargs["adgn_epsilon"],
                        gamma=kwargs["adgn_gamma"],
                        bias=kwargs["adgn_bias"],
                        gcn_conv=kwargs["adgn_gcn_norm"],
                        activ_fun=kwargs["adgn_activ_fun"],
                    )
                else:
                    # DISCRETE EDGE ONLY
                    return RelationalConv(
                        edge_size=edge_size,
                        conv_layer=AntiSymmetricConv,
                        in_channels=hidden_size,
                        num_iters=1,
                        epsilon=kwargs["adgn_epsilon"],
                        gamma=kwargs["adgn_gamma"],
                        bias=kwargs["adgn_bias"],
                        gcn_conv=kwargs["adgn_gcn_norm"],
                        activ_fun=kwargs["adgn_activ_fun"],
                    )

        def make_output(layer_id: int):
            if not kwargs["global_aggregation"]:
                if layer_id == -1:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear(node_size, node_size // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear(node_size // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
                else:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear(hidden_size, hidden_size // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear(hidden_size // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
            else:
                if layer_id == -1:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear((node_size * 3), (node_size * 3) // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear((node_size * 3) // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
                else:
                    return Sequential(
                        OrderedDict(
                            [
                                (
                                    "L1",
                                    Linear((hidden_size * 3), (hidden_size * 3) // 2),
                                ),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear((hidden_size * 3) // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )

        return make_hidden, make_output


class DGNGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Linear(node_size, hidden_size)
            else:
                conv_name = kwargs["conv_layer"]

                if edge_size == 0:
                    if conv_name == "GINConv":
                        mlp = Linear(hidden_size, hidden_size)
                        return EdgeFilterGINConv(nn=mlp, train_eps=True)

                    elif conv_name == "GCNConv":
                        return EdgeFilterGCNConv(
                            in_channels=hidden_size,
                            out_channels=hidden_size,
                            add_self_loops=False,
                        )
                    else:
                        raise NotImplementedError(
                            f"Conv layer not recognized: {conv_name}"
                        )
                else:
                    # DISCRETE EDGE ONLY
                    if conv_name == "GINConv":
                        mlp = Linear(hidden_size, hidden_size)
                        return RelationalConv(
                            edge_size=edge_size,
                            conv_layer=EdgeFilterGINConv,
                            nn=mlp,
                            train_eps=True,
                        )

                    elif conv_name == "GCNConv":
                        return RelationalConv(
                            edge_size=edge_size,
                            conv_layer=EdgeFilterGCNConv,
                            in_channels=hidden_size,
                            out_channels=hidden_size,
                            add_self_loops=False,
                        )
                    else:
                        raise NotImplementedError(
                            f"Conv layer not recognized: {conv_name}"
                        )

        def make_output(layer_id: int):
            if not kwargs["global_aggregation"]:
                if layer_id == -1:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear(node_size, node_size // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear(node_size // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
                else:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear(hidden_size, hidden_size // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear(hidden_size // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
            else:
                if layer_id == -1:
                    return Sequential(
                        OrderedDict(
                            [
                                ("L1", Linear((node_size * 3), (node_size * 3) // 2)),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear((node_size * 3) // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )
                else:
                    return Sequential(
                        OrderedDict(
                            [
                                (
                                    "L1",
                                    Linear((hidden_size * 3), (hidden_size * 3) // 2),
                                ),
                                ("LeakyReLU1", LeakyReLU()),
                                ("L2", Linear((hidden_size * 3) // 2, output_size)),
                                ("LeakyReLU2", LeakyReLU()),
                            ]
                        )
                    )

        return make_hidden, make_output


class LRGBGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        dropout = kwargs["dropout"]

        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Sequential(
                    Linear(node_size, hidden_size),
                    GELU(),
                    Linear(hidden_size, hidden_size),
                )
            else:
                conv_name = kwargs["conv_layer"]

                if conv_name == "GINConv":
                    mlp = Sequential(
                        Linear(hidden_size, hidden_size),
                        GELU(),
                        Linear(hidden_size, hidden_size),
                    )
                    return EdgeFilterGINConv(nn=mlp, train_eps=True)

                elif conv_name == "GCNConv":
                    return EdgeFilterGCNConv(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        add_self_loops=False,
                    )
                elif conv_name == "GINEConv":
                    mlp = Sequential(
                        Linear(hidden_size, hidden_size),
                        GELU(),
                        Linear(hidden_size, hidden_size),
                    )
                    return EdgeFilterGINEConv(
                        nn=mlp, train_eps=True, edge_dim=edge_size
                    )
                elif conv_name == "ResGatedGraphConv":
                    return EdgeFilterGatedGCNConv(hidden_size, hidden_size)
                else:
                    raise NotImplementedError(f"Conv layer not recognized: {conv_name}")

        def make_output(layer_id: int):
            if layer_id == -1:
                return Sequential(
                    Linear(node_size, hidden_size),
                    GELU(),
                    Dropout(dropout),
                    Linear(hidden_size, hidden_size),
                    GELU(),
                    Dropout(dropout),
                    Linear(hidden_size, output_size),
                )
            else:
                return Sequential(
                    Dropout(dropout),
                    Linear(hidden_size, hidden_size),
                    GELU(),
                    Dropout(dropout),
                    Linear(hidden_size, hidden_size),
                    GELU(),
                    Dropout(dropout),
                    Linear(hidden_size, output_size),
                )

        return make_hidden, make_output
