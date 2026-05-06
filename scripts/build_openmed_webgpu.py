# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "onnx",
#   "safetensors",
# ]
# ///
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper
from safetensors import safe_open


QK_SCALE = 1 / math.sqrt(math.sqrt(64))
BLOCK = 32


def bf16_to_fp32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.dtype("bfloat16"):
        return x.astype(np.float32)
    as_u16 = x.view(np.uint16).astype(np.uint32)
    return (as_u16 << 16).view(np.float32)


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint8)
    return values[..., 0::2] | (values[..., 1::2] << 4)


def quant_asym_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = rows.astype(np.float32)
    shape = rows.shape[:-1]
    blocks = rows.reshape(*shape, rows.shape[-1] // BLOCK, BLOCK)
    mn = blocks.min(axis=-1)
    mx = blocks.max(axis=-1)
    scale = (mx - mn) / 15.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    zp = np.clip(np.rint(-mn / scale), 0, 15).astype(np.uint8)
    q = np.clip(np.rint(blocks / scale[..., None] + zp[..., None]), 0, 15).astype(np.uint8)
    return pack_nibbles(q), scale, pack_nibbles(zp)


def quant_sym_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = rows.astype(np.float32)
    shape = rows.shape[:-1]
    blocks = rows.reshape(*shape, rows.shape[-1] // BLOCK, BLOCK)
    scale = blocks.max(axis=-1)
    scale = np.maximum(np.abs(blocks.min(axis=-1)), scale) / 8.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(blocks / scale[..., None]) + 8, 0, 15).astype(np.uint8)
    return pack_nibbles(q).reshape(*shape, rows.shape[-1] // 2), scale


def interleave_gate_up(x: np.ndarray) -> np.ndarray:
    gate, up = np.split(x, 2, axis=-1)
    out = np.empty_like(x)
    out[..., 0::2] = gate
    out[..., 1::2] = up
    return out


def set_init(model: onnx.ModelProto, name: str, arr: np.ndarray, data_type: int) -> None:
    init = next(i for i in model.graph.initializer if i.name == name)
    if data_type == TensorProto.FLOAT:
        arr = arr.astype(np.float32)
    elif data_type == TensorProto.UINT8:
        arr = arr.astype(np.uint8)
    init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr), name))


def replace_float_inits(model: onnx.ModelProto, weights) -> None:
    set_init(model, "model.layers.8.final_norm_layernorm.weight", bf16_to_fp32(weights.get_tensor("model.norm.weight")), TensorProto.FLOAT)
    for layer in range(8):
        prefix = f"model.layers.{layer}"
        set_init(model, f"{prefix}.input_layernorm.weight", bf16_to_fp32(weights.get_tensor(f"{prefix}.input_layernorm.weight")), TensorProto.FLOAT)
        set_init(model, f"{prefix}.post_attention_layernorm.weight", bf16_to_fp32(weights.get_tensor(f"{prefix}.post_attention_layernorm.weight")), TensorProto.FLOAT)
        set_init(model, f"{prefix}.attn.q_proj.Add.bias", bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.q_proj.bias")) * QK_SCALE, TensorProto.FLOAT)
        set_init(model, f"{prefix}.attn.o_proj.Add.bias", bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.o_proj.bias")), TensorProto.FLOAT)
        set_init(model, f"{prefix}.moe.experts.gate_up_proj.bias", interleave_gate_up(bf16_to_fp32(weights.get_tensor(f"{prefix}.mlp.experts.gate_up_proj_bias"))), TensorProto.FLOAT)
        set_init(model, f"{prefix}.moe.experts.down_proj.bias", bf16_to_fp32(weights.get_tensor(f"{prefix}.mlp.experts.down_proj_bias")), TensorProto.FLOAT)


def replace_quant_inits(model: onnx.ModelProto, weights) -> None:
    q, s, z = quant_asym_rows(bf16_to_fp32(weights.get_tensor("model.embed_tokens.weight")))
    set_init(model, "model_embed_tokens_weight_quant", q.reshape(q.shape[0], -1), TensorProto.UINT8)
    set_init(model, "model_embed_tokens_weight_scales", s, TensorProto.FLOAT)
    set_init(model, "model_embed_tokens_weight_zp", z, TensorProto.UINT8)

    score_q, score_s, score_z = quant_asym_rows(bf16_to_fp32(weights.get_tensor("score.weight")))
    set_init(model, "model_score_MatMul_weight_quant", score_q, TensorProto.UINT8)
    set_init(model, "model_score_MatMul_weight_scales", score_s, TensorProto.FLOAT)
    set_init(model, "model_score_MatMul_weight_zp", score_z, TensorProto.UINT8)

    for layer in range(8):
        prefix = f"model.layers.{layer}"
        quant_prefix = f"model_layers_{layer}"
        dense_weights = {
            "attn_q_proj_MatMul_weight": bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.q_proj.weight")).T * QK_SCALE,
            "attn_k_proj_MatMul_weight": bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.k_proj.weight")).T * QK_SCALE,
            "attn_v_proj_MatMul_weight": bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.v_proj.weight")).T,
            "attn_o_proj_MatMul_weight": bf16_to_fp32(weights.get_tensor(f"{prefix}.self_attn.o_proj.weight")).T,
            "moe_router_MatMul_weight": bf16_to_fp32(weights.get_tensor(f"{prefix}.mlp.router.weight")).T,
        }
        for suffix, weight in dense_weights.items():
            q, s, z = quant_asym_rows(weight.T)
            set_init(model, f"{quant_prefix}_{suffix}_quant", q, TensorProto.UINT8)
            set_init(model, f"{quant_prefix}_{suffix}_scales", s, TensorProto.FLOAT)
            set_init(model, f"{quant_prefix}_{suffix}_zp", z, TensorProto.UINT8)

        gate_up = interleave_gate_up(bf16_to_fp32(weights.get_tensor(f"{prefix}.mlp.experts.gate_up_proj")))
        q, s = quant_sym_rows(np.swapaxes(gate_up, 1, 2))
        set_init(model, f"{quant_prefix}_moe_experts_gate_up_proj_weight_quant", q, TensorProto.UINT8)
        set_init(model, f"{quant_prefix}_moe_experts_gate_up_proj_weight_scales", s, TensorProto.FLOAT)

        down = np.swapaxes(bf16_to_fp32(weights.get_tensor(f"{prefix}.mlp.experts.down_proj")), 1, 2)
        q, s = quant_sym_rows(down)
        set_init(model, f"{quant_prefix}_moe_experts_down_proj_weight_quant", q, TensorProto.UINT8)
        set_init(model, f"{quant_prefix}_moe_experts_down_proj_weight_scales", s, TensorProto.FLOAT)


def fix_score_head(model: onnx.ModelProto, weights) -> None:
    for node in model.graph.node:
        if node.name != "/model/score/MatMul_Quant":
            continue
        for attr in node.attribute:
            if attr.name == "N":
                attr.i = 221
        old_output = node.output[0]
        node.output[0] = "logits_internal"
        bias = bf16_to_fp32(weights.get_tensor("score.bias"))
        model.graph.initializer.append(numpy_helper.from_array(bias.astype(np.float32), "model.score.Add.bias"))
        model.graph.node.append(helper.make_node("Add", ["logits_internal", "model.score.Add.bias"], [old_output], name="/model/score/Add"))
        break
    else:
        raise RuntimeError("OpenAI q4 score node not found")

    dims = model.graph.output[0].type.tensor_type.shape.dim
    dims[2].dim_value = 221


def externalize_large_tensors(model: onnx.ModelProto, onnx_dir: Path, stem: str, shard_limit: int) -> int:
    for old in onnx_dir.glob(f"{stem}.onnx_data*"):
        old.unlink()

    shard = None
    shard_size = 0
    shard_count = 0

    def open_shard(index: int):
        location = f"{stem}.onnx_data" if index == 0 else f"{stem}.onnx_data_{index}"
        return location, open(onnx_dir / location, "wb")

    for init in model.graph.initializer:
        if len(init.raw_data) <= 1024:
            continue
        data = init.raw_data
        if shard is None or shard_size + len(data) > shard_limit:
            if shard is not None:
                shard.close()
            location, shard = open_shard(shard_count)
            shard_count += 1
            shard_size = 0
        offset = shard_size
        shard.write(data)
        shard_size += len(data)
        del init.external_data[:]
        init.data_location = TensorProto.EXTERNAL
        for key, value in (("location", location), ("offset", str(offset)), ("length", str(len(data)))):
            entry = init.external_data.add()
            entry.key = key
            entry.value = value
        init.ClearField("raw_data")

    if shard is not None:
        shard.close()
    return shard_count


def build(openmed_dir: Path, openai_dir: Path, out_dir: Path, shard_limit: int) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "onnx").mkdir(parents=True)

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(openmed_dir / name, out_dir / name)

    template = openai_dir / "onnx/model_q4.onnx"
    model = onnx.load(str(template), load_external_data=False)
    external_data_helper.load_external_data_for_model(model, str(template.parent))

    with safe_open(str(openmed_dir / "model.safetensors"), framework="np") as weights:
        replace_float_inits(model, weights)
        replace_quant_inits(model, weights)
        fix_score_head(model, weights)

    shard_count = externalize_large_tensors(model, out_dir / "onnx", "model_q4", shard_limit)
    onnx.save_model(model, str(out_dir / "onnx/model_q4.onnx"))

    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["transformers.js_config"] = {
        "use_external_data_format": {"model_q4.onnx": shard_count},
        "dtype": "q4",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Transformers.js WebGPU q4 artifact for OpenMed/privacy-filter-nemotron.")
    parser.add_argument("--openmed-dir", type=Path, required=True, help="Directory containing OpenMed/privacy-filter-nemotron files.")
    parser.add_argument("--openai-dir", type=Path, required=True, help="Directory containing openai/privacy-filter files, including onnx/model_q4.onnx.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for the WebGPU model artifact.")
    parser.add_argument("--shard-limit", type=int, default=450_000_000, help="Maximum external-data shard size in bytes.")
    args = parser.parse_args()

    build(args.openmed_dir, args.openai_dir, args.out_dir, args.shard_limit)
    print(args.out_dir)


if __name__ == "__main__":
    main()
