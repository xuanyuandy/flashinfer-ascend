"""AOT 子系统（设计文档 §7）：artifact 加载与校验。

M0 定义 manifest 数据结构与校验入口；产物发现、dlopen 与
``FLASHINFER_NPU_USE_AOT_ARTIFACT`` 灰度回退（§7.4）随首批 AOT
产物（M2）接入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class ArtifactVerificationError(RuntimeError):
    """manifest 校验失败（§7.4：显式报错或按策略转 JIT 重编译）。"""


@dataclass(frozen=True)
class ArtifactManifest:
    """产物身份声明：加载前校验，确认产物与当前环境匹配（§11）。"""

    kernel_id: str
    soc_models: Tuple[str, ...]
    toolkit_version: str
    artifact_schema: int = 1

    def verify(self, soc_model: str, toolkit_version: str) -> None:
        if self.soc_models and soc_model not in self.soc_models:
            raise ArtifactVerificationError(
                "产物 %s 不支持 SoC %s（支持集 %s）"
                % (self.kernel_id, soc_model, sorted(self.soc_models))
            )
        if self.toolkit_version != toolkit_version:
            raise ArtifactVerificationError(
                "产物 %s 由 toolkit %s 构建，当前为 %s"
                % (self.kernel_id, self.toolkit_version, toolkit_version)
            )
