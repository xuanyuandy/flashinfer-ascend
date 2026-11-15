# ABI 与冻结兼容规则

面向原生后端开发者：本页说明接口序列化、二进制、能力和冻结测试的边界，避免用修改指纹代替兼容实现。

## 身份分层

公共 schema 定义计算与输出；NativeBinding 定义原生 Tensor/属性顺序；recipe/spec 绑定构建输入、能力、ABI 和环境；包 manifest 绑定实际字节。这些对象独立校验，不能互相替代。

直接算子的 `domain="operator"` 绑定与 AttentionPlan 的 profile/POD ABI 是不同协议。已有 Host POD/metadata 兼容 reader 不意味着新生产算子要使用该协议；公开原生接入使用 v2 op-package。

## 维护规则

- 修改 ABI、类型组合、shape、能力、CMake 或源码后重新生成并冻结构建身份。
- 有产物或消费者依赖的旧协议保持原有校验，不能从新版本“继承”更多参数或能力。
- 冻结常量和精确登记集独立于被测生成器；不把当前运行输出直接作为测试答案。
- 需要改变冻结常量、放宽校验或删除护栏时，先准备具体差异与影响分析并取得明确授权。
- 可选 Tensor、属性、存储位解释、workspace 与同步协议要同时覆盖声明、包验证、适配和测试。

## 现有冻结依据

当前 binary/error ABI 与 metadata wire 的测试值沿用仓库已记录的重新冻结决定。该决定的理由是早期 error ABI 序列化无法完整恢复，漂移传播至两个 binary 指纹和 wire header；重新冻结前的检索没有发现活产物依赖旧值。

这项决定不证明新旧语义完全等价，也不允许再次自动重冻。若取得原始定义或发现旧消费者，需要重新评估兼容性和迁移方案。布局尺寸与字段护栏仍由测试保持。

| 冻结对象 | 当前检查位置 |
| --- | --- |
| binary ABI v1 与 error ABI | [runtime ABI 测试](../../tests/test_runtime_abi.py) |
| 量化 KV binary ABI v2 与布局 | [量化启动 ABI 测试](../../tests/test_attention_quant_launch_abi.py) |
| metadata wire 长度、布局与序列化身份 | [metadata wire 测试](../../tests/test_attention_metadata_wire.py) |
| 旧原生 recipe 与新版本边界 | [原生契约测试](../../tests/test_ascendc_contract_v2.py) |

测试中保留的旧文档路径均指向本规则。历史过程日志不作为当前运行结果发布；当前修改不得据这些历史结果宣称新的设备、图或性能验证。
