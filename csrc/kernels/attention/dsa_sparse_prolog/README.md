# DsaSparseProlog

DSA 稀疏路径 prolog（吸收模式）。Q 路径做 `weight_dq`、RMSNorm、`weight_uq_qr`、scale 和 RoPE；KV 路径做 `weight_dkv_kr`、split、RMSNorm、scale 和 RoPE。同时输出反向中间量。

A2（Ascend910B）和 A3（Ascend910C）共用这份 arch22 实现。算子身份保持 `DsaSparseProlog`，设备入口是 `dsa_sparse_prolog`。

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build \
  --kernel-id ascendc/attention/dsa_sparse_prolog/native/Ascend910B \
  --output-dir build/dsa-sparse-prolog-aot
```

A3 把 kernel id 换成 `ascendc/attention/dsa_sparse_prolog/native/Ascend910C`。
