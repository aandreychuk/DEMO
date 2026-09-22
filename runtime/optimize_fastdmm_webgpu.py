#!/usr/bin/env python3
"""Fuse FastDMM normalization and attention blocks for ONNX Runtime WebGPU."""

from __future__ import annotations

from pathlib import Path
import collections

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxconverter_common import float16

from runtime.convert_fastdmm_fp16 import validation_inputs


root = Path(__file__).resolve().parent.parent
source_path = root / "public/runtime/fastdmm-0.8m.onnx"
target_path = root / "public/runtime/fastdmm-0.8m-webgpu.onnx"
fp16_path = root / "public/runtime/fastdmm-0.8m-webgpu-fp16.onnx"


def graph_maps(nodes):
    producer = {output: node for node in nodes for output in node.output if output}
    consumers = collections.defaultdict(list)
    for node in nodes:
        for name in node.input:
            if name:
                consumers[name].append(node)
    return producer, consumers


model = onnx.load(source_path)
inferred_model = onnx.shape_inference.infer_shapes(model)


def tensor_shape(value_info):
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


shapes = {
    value.name: tensor_shape(value)
    for value in [
        *inferred_model.graph.input,
        *inferred_model.graph.value_info,
        *inferred_model.graph.output,
    ]
}
nodes = list(model.graph.node)
producer, consumers = graph_maps(nodes)
initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}

remove = set()
replacement = collections.defaultdict(list)
fused_norms = 0
for mean in nodes:
    if mean.op_type != "ReduceMean" or len(mean.output) != 1:
        continue
    power = producer.get(mean.input[0])
    if power is None or power.op_type != "Pow" or len(power.input) != 2:
        continue
    exponent = initializers.get(power.input[1])
    if exponent is None or not np.allclose(exponent, 2.0):
        continue
    root_name = power.input[0]
    if len(consumers[mean.output[0]]) != 1:
        continue
    add = consumers[mean.output[0]][0]
    if add.op_type != "Add":
        continue
    epsilon_name = next((name for name in add.input if name in initializers), None)
    epsilon = initializers.get(epsilon_name) if epsilon_name else None
    if epsilon is None or epsilon.size != 1 or len(consumers[add.output[0]]) != 1:
        continue
    sqrt = consumers[add.output[0]][0]
    if sqrt.op_type != "Sqrt" or len(consumers[sqrt.output[0]]) != 1:
        continue
    reciprocal = consumers[sqrt.output[0]][0]
    if reciprocal.op_type != "Reciprocal" or len(consumers[reciprocal.output[0]]) != 1:
        continue
    inner = consumers[reciprocal.output[0]][0]
    if inner.op_type != "Mul" or root_name not in inner.input:
        continue
    outer_nodes = consumers[inner.output[0]]
    if not outer_nodes:
        continue
    scales: list[tuple[onnx.NodeProto, str]] = []
    for outer in outer_nodes:
        if outer.op_type != "Mul" or len(outer.input) != 2:
            scales = []
            break
        scale_name = outer.input[0] if outer.input[1] == inner.output[0] else outer.input[1]
        if scale_name not in initializers:
            scales = []
            break
        scales.append((outer, scale_name))
    if not scales:
        continue
    chain = [power, mean, add, sqrt, reciprocal, inner]
    if any(len(consumers[node.output[0]]) != 1 for node in chain[:-1]):
        continue
    remove.update(id(node) for node in [*chain, *(outer for outer, _ in scales)])
    for outer, scale_name in scales:
        replacement[id(outer)].append(
            helper.make_node(
                "SimplifiedLayerNormalization",
                [root_name, scale_name],
                list(outer.output),
                name=f"FusedRMSNorm_{fused_norms}",
                axis=-1,
                epsilon=float(epsilon.reshape(-1)[0]),
                stash_type=1,
            )
        )
        fused_norms += 1

nodes_after_rms = []
for node in nodes:
    if id(node) in replacement:
        nodes_after_rms.extend(replacement[id(node)])
    elif id(node) not in remove:
        nodes_after_rms.append(node)

producer, consumers = graph_maps(nodes_after_rms)
remove = set()
replacement = collections.defaultdict(list)
attention_zeros: dict[tuple[int, ...], str] = {}
fused_attentions = 0


def zero_initializer(shape: list[int]) -> str:
    key = tuple(shape)
    if key not in attention_zeros:
        name = f"webgpu_attention_zero_{len(attention_zeros)}"
        model.graph.initializer.append(numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name=name))
        attention_zeros[key] = name
    return attention_zeros[key]


for softmax in nodes_after_rms:
    if softmax.op_type != "Softmax" or shapes.get(softmax.output[0], [])[-1:] == [5]:
        continue
    masked = producer.get(softmax.input[0])
    if masked is None or masked.op_type != "Where" or len(masked.input) != 3:
        continue
    scaled = producer.get(masked.input[2])
    if scaled is None or scaled.op_type != "Mul":
        continue
    qk = next((producer.get(name) for name in scaled.input if producer.get(name) is not None), None)
    if qk is None or qk.op_type != "MatMul":
        continue
    q_transpose = producer.get(qk.input[0])
    k_transpose = producer.get(qk.input[1])
    if q_transpose is None or q_transpose.op_type != "Transpose" or k_transpose is None or k_transpose.op_type != "Transpose":
        continue
    score_consumers = consumers[softmax.output[0]]
    if len(score_consumers) != 1 or score_consumers[0].op_type != "MatMul":
        continue
    score = score_consumers[0]
    output_transpose_consumers = consumers[score.output[0]]
    if len(output_transpose_consumers) != 1 or output_transpose_consumers[0].op_type != "Transpose":
        continue
    output_transpose = output_transpose_consumers[0]
    output_reshape_consumers = consumers[output_transpose.output[0]]
    if len(output_reshape_consumers) != 1 or output_reshape_consumers[0].op_type != "Reshape":
        continue
    output_reshape = output_reshape_consumers[0]

    query_base = q_transpose.input[0]
    key_base = k_transpose.input[0]
    value_base = score.input[1]
    value_transpose = producer.get(value_base)
    if value_transpose is not None and value_transpose.op_type == "Transpose":
        value_base = value_transpose.input[0]
    else:
        value_transpose = None

    query_shape = shapes.get(query_base)
    key_shape = shapes.get(key_base)
    value_shape = shapes.get(value_base)
    score_shape = shapes.get(masked.output[0])
    if not query_shape or len(query_shape) != 4 or not key_shape or not value_shape or not score_shape:
        continue
    batch_size, query_length, num_heads, head_size = query_shape
    query_flat_shape = [batch_size, query_length, num_heads * head_size]
    query_shape_name = f"webgpu_mha_query_shape_{fused_attentions}"
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(query_flat_shape, dtype=np.int64), name=query_shape_name)
    )
    query_flat = f"webgpu_mha_query_{fused_attentions}"
    new_nodes = [
        helper.make_node(
            "Reshape",
            [query_base, query_shape_name],
            [query_flat],
            name=f"WebGpuMhaQueryReshape_{fused_attentions}",
        )
    ]

    key_input = key_base
    if len(key_shape) == 4 and key_shape[1] != num_heads:
        key_flat_shape = [key_shape[0], key_shape[1], key_shape[2] * key_shape[3]]
        key_shape_name = f"webgpu_mha_key_shape_{fused_attentions}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(key_flat_shape, dtype=np.int64), name=key_shape_name)
        )
        key_input = f"webgpu_mha_key_{fused_attentions}"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [key_base, key_shape_name],
                [key_input],
                name=f"WebGpuMhaKeyReshape_{fused_attentions}",
            )
        )

    value_input = value_base
    if len(value_shape) == 4 and value_shape[1] != num_heads:
        value_flat_shape = [value_shape[0], value_shape[1], value_shape[2] * value_shape[3]]
        value_shape_name = f"webgpu_mha_value_shape_{fused_attentions}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(value_flat_shape, dtype=np.int64), name=value_shape_name)
        )
        value_input = f"webgpu_mha_value_{fused_attentions}"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [value_base, value_shape_name],
                [value_input],
                name=f"WebGpuMhaValueReshape_{fused_attentions}",
            )
        )

    new_nodes.append(
        helper.make_node(
            "MultiHeadAttention",
            [query_flat, key_input, value_input, "", "", masked.output[0]],
            [output_reshape.output[0]],
            name=f"WebGpuMha_{fused_attentions}",
            domain="com.microsoft",
            num_heads=num_heads,
            scale=1.0 / np.sqrt(head_size),
        )
    )
    replacement[id(masked)].append(
        helper.make_node(
            "Where",
            [masked.input[0], masked.input[1], zero_initializer(score_shape)],
            list(masked.output),
            name=f"WebGpuMhaBias_{fused_attentions}",
        )
    )
    replacement[id(output_reshape)].extend(new_nodes)
    remove.update(
        id(node)
        for node in [
            q_transpose,
            k_transpose,
            qk,
            scaled,
            masked,
            softmax,
            score,
            output_transpose,
            output_reshape,
            *([value_transpose] if value_transpose is not None else []),
        ]
    )
    fused_attentions += 1

final_nodes = []
for node in nodes_after_rms:
    if id(node) in replacement:
        final_nodes.extend(replacement[id(node)])
    elif id(node) not in remove:
        final_nodes.append(node)

del model.graph.node[:]
model.graph.node.extend(final_nodes)
if not any(item.domain == "com.microsoft" for item in model.opset_import):
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
onnx.save(model, target_path)
print(
    "fused norms", fused_norms,
    "fused attentions", fused_attentions,
    "nodes", len(nodes), "->", len(final_nodes),
    "bytes", target_path.stat().st_size,
)

fp16_model = float16.convert_float_to_float16(
    model,
    min_positive_val=5.9604645e-8,
    max_finite_val=65504.0,
    keep_io_types=True,
)
onnx.save(fp16_model, fp16_path)
print("fp16 bytes", fp16_path.stat().st_size)

options = ort.SessionOptions()
options.log_severity_level = 3
reference_session = ort.InferenceSession(str(source_path), sess_options=options, providers=["CPUExecutionProvider"])
fused_session = ort.InferenceSession(str(target_path), sess_options=options, providers=["CPUExecutionProvider"])
fp16_session = ort.InferenceSession(str(fp16_path), sess_options=options, providers=["CPUExecutionProvider"])
for index, feeds in enumerate(validation_inputs(100)):
    reference = reference_session.run(None, feeds)[0]
    actual = fused_session.run(None, feeds)[0]
    fp16 = fp16_session.run(None, feeds)[0]
    print(
        index,
        np.max(np.abs(actual - reference)),
        np.mean(actual.argmax(-1) == reference.argmax(-1)),
        np.max(np.abs(fp16 - reference)),
        np.mean(fp16.argmax(-1) == reference.argmax(-1)),
    )
