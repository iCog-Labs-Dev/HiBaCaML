"""Static graph construction for HiBaCaML."""

from __future__ import annotations

from typing import Dict, List

from fabricpc.core.topology import Edge, GraphNamespace
from fabricpc.graph_assembly import graph, TaskMap
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.core.activations import IdentityActivation, SoftmaxActivation, TanhActivation
from fabricpc.core.inference import InferenceBase
from fabricpc.graph_initialization.state_initializer import StateInitBase
from hibacaml.config import HiBaCaMLConfig
from hibacaml.nodes import (
    ComposerStage2Node,
    ElementwiseGateNode,
    PatchTokenizerNode,
    ScaledAddNode,
    ShellBankInputNode,
    ShellBankRecurrentNode,
    ShellBankResidualNode,
    WeightedCrossEntropyEnergy,
)
from hibacaml.types import (
    ColumnCertificate,
    PersistentHiBaCaMLState,
    ShellStats
    )



def create_hibacaml_structure(
    cfg: HiBaCaMLConfig,
    inference: InferenceBase,
    graph_state_initializer: StateInitBase,
):
    """Create the full static column-bank graph.

    Any ``StateInitBase`` is accepted here so the builder stays independent of
    training concerns, but both HiBaCaML runners require ``FeedforwardStateInit``.
    """
    nodes = []
    edges = []
    column_metadata: List[Dict[str, str]] = []
    cert_input_names: Dict[int, str] = {}
    feature_gate_names: Dict[int, str] = {}
    logit_gate_names: Dict[int, str] = {}
    b_micro_names: Dict[int, str] = {}
    k_micro_names: Dict[int, str] = {}
    k_micro_depth_names: Dict[int, tuple[str, ...]] = {}
    l_micro_names: Dict[int, str] = {}

    image_input = IdentityNode(shape=cfg.input_shape, name="image_input")
    support_mask = IdentityNode(shape=(cfg.support_vector_dim,), name="support_mask")
    task_query = IdentityNode(shape=(cfg.composer.query_dim,), name="task_query")
    patch_tokens = PatchTokenizerNode(
        shape=(cfg.num_patches, cfg.patch_token_dim),
        name="patch_tokens",
        patch_size=cfg.patch_size,
        patch_embed_dim=cfg.patch_embed_dim,
        coord_dim=cfg.patch_coord_dim,
    )
    nodes.extend([image_input, support_mask, task_query, patch_tokens])
    edges.append(Edge(source=image_input, target=patch_tokens.slot("in")))

    for column_index in range(cfg.column_pool.total_columns):
        with GraphNamespace(f"col{column_index}"):
            cert_input = IdentityNode(
                shape=(cfg.composer_cert_dim,),
                name="cert_input",
            )
            token_gate = ElementwiseGateNode(
                shape=(cfg.num_patches, cfg.patch_token_dim),
                name="token_gate",
                gate_index=column_index,
            )
            b_micro = ShellBankInputNode(
                shape=(cfg.num_patches, cfg.column_pool.shell_output_dim),
                name="b_micro",
                activation=TanhActivation(),
                kernel_dim=cfg.column_pool.memory_dim,
                shell_sizes=cfg.column_pool.shell_sizes,
            )
            kernel_nodes = []
            for depth_idx in range(max(1, cfg.column_pool.kernel_depth)):
                kernel_nodes.append(
                    ShellBankRecurrentNode(
                        shape=(cfg.num_patches, cfg.column_pool.shell_output_dim),
                        name="k_micro" if depth_idx == 0 else f"k_micro_{depth_idx}",
                        activation=TanhActivation(),
                        kernel_dim=cfg.column_pool.memory_dim,
                        shell_sizes=cfg.column_pool.shell_sizes,
                    )
                )
            l_micro = ShellBankResidualNode(
                shape=(cfg.num_patches, cfg.column_pool.shell_output_dim),
                name="l_micro",
                activation=TanhActivation(),
                kernel_dim=cfg.column_pool.memory_dim,
                shell_sizes=cfg.column_pool.shell_sizes,
            )
            feature_pool = Linear(
                shape=(cfg.composer.hidden_dim,),
                name="feature_pool",
                activation=TanhActivation(),
                flatten_input=True,
            )
            feature_gate = ElementwiseGateNode(
                shape=(cfg.composer.hidden_dim,),
                name="feature_gate",
                gate_index=column_index,
            )
            column_logits = Linear(
                shape=(cfg.output_dim,),
                name="column_logits",
                activation=IdentityActivation(),
            )
            logit_gate = ElementwiseGateNode(
                shape=(cfg.output_dim,),
                name="logit_gate",
                gate_index=column_index,
            )
            nodes.extend(
                [
                    cert_input,
                    token_gate,
                    b_micro,
                    *kernel_nodes,
                    l_micro,
                    feature_pool,
                    feature_gate,
                    column_logits,
                    logit_gate,
                ]
            )
            edges.extend(
                [
                    Edge(source=patch_tokens, target=token_gate.slot("value")),
                    Edge(source=support_mask, target=token_gate.slot("gate")),
                    Edge(source=token_gate, target=b_micro.slot("in")),
                    Edge(source=b_micro, target=l_micro.slot("skip")),
                    Edge(source=b_micro, target=feature_pool.slot("in")),
                    Edge(source=l_micro, target=feature_pool.slot("in")),
                    Edge(source=feature_pool, target=feature_gate.slot("value")),
                    Edge(source=support_mask, target=feature_gate.slot("gate")),
                    Edge(source=feature_pool, target=column_logits.slot("in")),
                    Edge(source=column_logits, target=logit_gate.slot("value")),
                    Edge(source=support_mask, target=logit_gate.slot("gate")),
                ]
            )
            previous_kernel_source = b_micro
            for kernel_node in kernel_nodes:
                edges.append(Edge(source=previous_kernel_source, target=kernel_node.slot("in")))
                edges.append(Edge(source=kernel_node, target=l_micro.slot("skip")))
                edges.append(Edge(source=kernel_node, target=feature_pool.slot("in")))
                previous_kernel_source = kernel_node
            feature_gate_names[column_index] = feature_gate.name
            logit_gate_names[column_index] = logit_gate.name
            cert_input_names[column_index] = cert_input.name
            b_micro_names[column_index] = b_micro.name
            k_micro_names[column_index] = kernel_nodes[-1].name
            k_micro_depth_names[column_index] = tuple(node.name for node in kernel_nodes)
            l_micro_names[column_index] = l_micro.name
            column_metadata.append(
                {
                    "cert_input": cert_input.name,
                    "token_gate": token_gate.name,
                    "b_micro": b_micro.name,
                    "k_micro": kernel_nodes[-1].name,
                    "k_micro_depth": tuple(node.name for node in kernel_nodes),
                    "l_micro": l_micro.name,
                    "feature_pool": feature_pool.name,
                    "feature_gate": feature_gate.name,
                    "column_logits": column_logits.name,
                    "logit_gate": logit_gate.name,
                }
            )

    for target_index in range(cfg.column_pool.total_columns):
        l_node = next(node for node in nodes if node.name == l_micro_names[target_index])
        for source_index in range(cfg.column_pool.total_columns):
            if source_index == target_index:
                continue
            k_node = next(node for node in nodes if node.name == k_micro_names[source_index])
            edges.append(Edge(source=k_node, target=l_node.slot("in")))

    stage1_logits = IdentityNode(
        shape=(cfg.output_dim,),
        name="stage1_logits",
        scale=1.0 / cfg.column_pool.active_support_size,
    )
    active_feature_summary = IdentityNode(
        shape=(cfg.composer.hidden_dim,),
        name="active_feature_summary",
        scale=1.0 / cfg.column_pool.active_support_size,
    )
    composer2 = ComposerStage2Node(
        shape=(cfg.output_dim,),
        name="composer2",
        hidden_dim=cfg.composer.hidden_dim,
        gate_temp=cfg.composer.gate_temp,
        prior_logit_scale=cfg.composer.prior_logit_scale,
        prior_mix_scale=cfg.composer.prior_mix_scale,
        residual_gate_scale=cfg.composer.residual_gate_scale,
        query_score_scale=cfg.composer.query_score_scale,
        prior_kl_weight=cfg.composer.prior_kl_weight,
        gate_entropy_ceiling_frac=cfg.composer.gate_entropy_ceiling_frac,
        gate_entropy_ceiling_weight=cfg.composer.gate_entropy_ceiling_weight,
        gate_dev_floor=cfg.composer.gate_dev_floor,
        gate_dev_weight=cfg.composer.gate_dev_weight,
        topk=cfg.composer.topk,
    )
    final_output = ScaledAddNode(
        shape=(cfg.output_dim,),
        name="final_output",
        correction_scale=cfg.composer.scale,
        activation=SoftmaxActivation(),
        energy=WeightedCrossEntropyEnergy(weight=1.0),
    )
    hier_mid = Linear(
        shape=(cfg.hierarchy.mid_targets, cfg.output_dim),
        name="hier_mid",
        activation=SoftmaxActivation(),
        energy=WeightedCrossEntropyEnergy(weight=cfg.hierarchy.mid_loss_weight),
        flatten_input=True,
    )
    hier_global = Linear(
        shape=(cfg.output_dim,),
        name="hier_global",
        activation=SoftmaxActivation(),
        energy=WeightedCrossEntropyEnergy(weight=cfg.hierarchy.global_loss_weight),
    )
    nodes.extend([stage1_logits, active_feature_summary, composer2, final_output, hier_mid, hier_global])

    for column_index in range(cfg.column_pool.total_columns):
        feature_gate_name = feature_gate_names[column_index]
        logit_gate_name = logit_gate_names[column_index]
        cert_input_name = cert_input_names[column_index]
        feature_gate_node = next(node for node in nodes if node.name == feature_gate_name)
        logit_gate_node = next(node for node in nodes if node.name == logit_gate_name)
        cert_input_node = next(node for node in nodes if node.name == cert_input_name)
        edges.append(Edge(source=logit_gate_node, target=stage1_logits.slot("in")))
        edges.append(Edge(source=feature_gate_node, target=active_feature_summary.slot("in")))
        edges.append(Edge(source=feature_gate_node, target=composer2.slot("feature")))
        edges.append(Edge(source=cert_input_node, target=composer2.slot("cert")))

    edges.extend(
        [
            Edge(source=task_query, target=composer2.slot("query")),
            Edge(source=stage1_logits, target=final_output.slot("base")),
            Edge(source=composer2, target=final_output.slot("correction")),
            Edge(source=active_feature_summary, target=hier_mid.slot("in")),
            Edge(source=active_feature_summary, target=hier_global.slot("in")),
        ]
    )

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(
            x=image_input,
            y=final_output,
            hier_mid=hier_mid,
            hier_global=hier_global,
        ),
        inference=inference,
        graph_state_initializer=graph_state_initializer,
    )

    hibacaml_meta = {
        "cfg": cfg,
        "support_mask_node": support_mask.name,
        "task_query_node": task_query.name,
        "patch_tokens_node": patch_tokens.name,
        "stage1_logits_node": stage1_logits.name,
        "active_feature_summary_node": active_feature_summary.name,
        "composer2_node": composer2.name,
        "final_output_node": final_output.name,
        "column_nodes": tuple(column_metadata),
        "b_micro_names": b_micro_names,
        "k_micro_names": k_micro_names,
        "k_micro_depth_names": k_micro_depth_names,
        "l_micro_names": l_micro_names,
        "cert_input_names": cert_input_names,
        "feature_gate_names": feature_gate_names,
        "logit_gate_names": logit_gate_names,
    }
    return structure._replace(
        config={
            **structure.config,
            "hibacaml": hibacaml_meta,
        }
    )


def initialize_hibacaml_state(
    params,
    cfg: HiBaCaMLConfig,
) -> PersistentHiBaCaMLState:
    """Create persistent state container for a new HiBaCaML run."""
    shell_stats = {
        idx: ShellStats(
            activation_ema={
                "kernel": 0.0,
                "tier1": 0.0,
                "tier2": 0.0,
                "tier3": 0.0,
            },
            task_variance_ema={
                "kernel": 0.0,
                "tier1": 0.0,
                "tier2": 0.0,
                "tier3": 0.0,
            },
        )
        for idx in range(cfg.column_pool.total_columns)
    }
    certificates = {
        idx: ColumnCertificate(
            column_index=idx,
            q_mean=0.5,
            prec_mean=1.0,
            pred_mean=0.0,
            live_frac=1.0 if idx in cfg.column_pool.shared_indices else 0.0,
            tier_q=(0.5, 0.5, 0.5),
            tier_occ=(0.0, 0.0, 0.0),
            shared_abstraction_mass=0.0,
            specificity_load=0.0,
            demotion_pressure=0.0,
            saturation=0.0,
            similarity_signature=tuple(0.0 for _ in range(cfg.column_pool.total_columns)),
        )
        for idx in range(cfg.column_pool.total_columns)
    }
    return PersistentHiBaCaMLState(
        shell_stats=shell_stats,
        certificates=certificates,
        params=params,
    )
