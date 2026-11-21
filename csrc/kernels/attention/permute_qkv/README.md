# PermuteQkv（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| Atlas A3系列产品              |    √     |
| Atlas A2系列产品              |    √     |

## 功能说明

`PermuteQkv` 将 query/key/value 按 KV group 做头重排（N->SPG），把 `[s_c, b, q_head_num, h]`
与 `[s_c, b, kv_head_num, h]` 的 q/k/v 拼接后重排为 `[total_head, s_c, b, h]`，用于后续 MoE/序列并行分组。

## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                  | 数据类型       | 格式 |
| ---------- | -------------- | ------------------------------------- | -------------- | :--: |
| q          | 输入           | `[s_c, b, q_head_num, h]`             | BF16 / FLOAT16 | ND   |
| k          | 输入           | `[s_c, b, kv_head_num, h]`            | BF16 / FLOAT16 | ND   |
| v          | 输入           | `[s_c, b, kv_head_num, h]`            | BF16 / FLOAT16 | ND   |
| split_size | 必选属性       | 分组数（SPG）                         | INT            | -    |
| output     | 输出           | `[q_head_num + 2*kv_head_num, s_c, b, h]` | BF16 / FLOAT16 | ND |

## FlashInfer 源码构建

在仓库根目录按 [README 的环境准备和编译步骤](../../../../README.md#原生算子编译与精度验证) 构建设备包与 ACLNN 桥。A2 使用 `ascendc/attention/permute_qkv/native/Ascend910B`，A3 使用 `Ascend910C` 后缀。用 `python -B tools/run_regression.py --profile full -k "permute_qkv and not unpermute_qkv" tests/operators` 运行公共接口精度和边界用例。

## 上游 ops-transformer 工程的构建命令

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=permute_qkv -j4
```
