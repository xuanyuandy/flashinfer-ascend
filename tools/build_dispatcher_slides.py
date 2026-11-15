#!/usr/bin/env python3
"""Build editable architecture slides from the current routing design.

Documentation only. The PPTX contains native shapes, text and connectors.
Source references and execution boundaries live in the speaker notes, leaving
the slides readable without source listings. No execution runner is invoked.
The generator maintains one four-page edition. Routing examples are checked
against the metadata-only demo at build time; no kernel is compiled or launched.

Dependencies: python-pptx and matplotlib. Optional isolated dependencies may
be installed in build/routing-slides/python-deps. Use LibreOffice to export
the generated PPTX for the final PDF preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "build/routing-slides/python-deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties


WIDTH, HEIGHT = 16, 9
INK, MUTED = "172E46", "5C6C7E"
BLUE, TEAL, AMBER, PURPLE = "275FA8", "087F83", "A9650D", "7552AD"
PALE, BORDER, WHITE = "F4F7FB", "DCE4EE", "FFFFFF"
FILL = {BLUE: "EDF3FC", TEAL: "EDF8F6", AMBER: "FFF6E8", PURPLE: "F4F0FB"}
FONT = "Noto Sans CJK SC"
DEFAULT_FONT = ROOT / "build/matmul/diagram-assets/NotoSansCJKsc-Regular.otf"
STEM = "dispatcher"
ARTIFACTS = ROOT / "build/dispatcher-docs"
SOURCE_PATHS = (
    "flashinfer/gemm.py", "flashinfer/_matmul.py", "flashinfer/aclnn/_ops.py",
    "flashinfer/runtime/signature.py", "flashinfer/runtime/hardware.py",
    "flashinfer/runtime/schema.py", "flashinfer/runtime/registry.py",
    "flashinfer/runtime/dispatch.py", "flashinfer/runtime/runner.py",
    "flashinfer/cann/gemm/matmul/implementation.py", "flashinfer/cann/gemm/matmul/runner.py",
    "flashinfer/ascendc/__init__.py", "flashinfer/ascendc/gemm/matmul/implementation.py",
    "flashinfer/ascendc/gemm/matmul/runner.py", "flashinfer/triton/gemm/matmul/implementation.py",
    "flashinfer/triton/gemm/matmul/runner.py",
    "build/generated/flashinfer/data/kernels/registry.json", "build/generated/flashinfer/data/kernels/ascendc_specs.json",
    "build/generated/csrc/recipes/matmul.json",
    "flashinfer/jit/runtime.py", "tools/routing_capability_demo.py",
    "flashinfer/jit/tensor_adapter.py", "flashinfer/aclnn/_native.py",
)


def color(value):
    return RGBColor.from_string(value)


class Deck:
    def __init__(self):
        self.total = 4
        self.prs = Presentation()
        self.prs.slide_width = Inches(WIDTH)
        self.prs.slide_height = Inches(HEIGHT)
        self.prs.core_properties.title = "FlashInfer-Ascend：多后端 Dispatcher 核心设计"
        self.prs.core_properties.subject = "模块职责、Matmul 三后端链路、注册与 kernel launch"
        self.prs.core_properties.author = "FlashInfer-Ascend"
        self.prs.core_properties.keywords = "dispatcher, registry, matmul, CANN, AOT, Triton"
        self.texts = []

    def slide(self, number, title, subtitle):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = color(WHITE)
        self.box(slide, .65, .42, .07, .22, BLUE)
        self.text(slide, .86, .38, 10, .3, "FLASHINFER · ASCEND   /   MULTI-BACKEND ROUTING", 11.5, MUTED)
        self.text(slide, .65, .87, 14.7, .65, title, 30, INK, bold=True)
        self.text(slide, .65, 1.59, 14.7, .4, subtitle, 17.5, MUTED)
        self.line(slide, .65, 8.55, 15.35, 8.55, BORDER, width=.7)
        self.text(slide, .65, 8.67, 12, .23, "FlashInfer-Ascend  ·  多后端路由核心设计", 10.5, MUTED)
        self.text(slide, 14.2, 8.63, 1.15, .3, f"{number} / {self.total}", 12, MUTED, align="right")
        return slide

    def box(self, slide, x, y, w, h, fill=WHITE, stroke=None, rounded=False):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        if rounded:
            shape.adjustments[0] = .10
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(fill)
        if stroke:
            shape.line.color.rgb = color(stroke)
            shape.line.width = Pt(.8)
        else:
            shape.line.fill.background()
        # Override the template's shadow effect with an explicitly flat style.
        shape._element.spPr.append(OxmlElement("a:effectLst"))
        style = shape._element.find(qn("p:style"))
        if style is not None:
            shape._element.remove(style)
        return shape

    def text(self, slide, x, y, w, h, content, size=18, ink=INK, bold=False,
             align="left", middle=False):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = shape.text_frame
        frame.clear()
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE if middle else MSO_ANCHOR.TOP
        frame._txBody.bodyPr.set("anchorCtr", "0")
        for index, line in enumerate(content.split("\n")):
            para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            para.text = line
            para.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                              "right": PP_ALIGN.RIGHT}[align]
            para.line_spacing = Pt(size * 1.27)
            para.space_before = Pt(0)
            para.space_after = Pt(0)
            para.font.name = FONT
            para.font.size = Pt(size)
            para.font.bold = bold
            para.font.color.rgb = color(ink)
            # PowerPoint's East Asian fallback must use the same face.
            props = para._p.get_or_add_pPr().get_or_add_defRPr()
            for tag in ("a:ea", "a:cs"):
                face = OxmlElement(tag)
                face.set("typeface", FONT)
                props.append(face)
        self.texts.append(dict(slide=len(self.prs.slides), x=x, y=y, w=w, h=h,
                               content=content, size=size, bold=bold))
        return shape

    def line(self, slide, x1, y1, x2, y2, ink=BORDER, width=1.5, arrow=False):
        shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                           Inches(x2), Inches(y2))
        shape.line.color.rgb = color(ink)
        shape.line.width = Pt(width)
        shape._element.spPr.append(OxmlElement("a:effectLst"))
        style = shape._element.find(qn("p:style"))
        if style is not None:
            shape._element.remove(style)
        if arrow:
            end = OxmlElement("a:tailEnd")
            end.set("type", "triangle")
            end.set("w", "med")
            end.set("len", "med")
            shape.line._get_or_add_ln().append(end)
        return shape

    def card(self, slide, x, y, w, h, title, body, accent=BLUE, size=17, title_size=21):
        self.box(slide, x, y, w, h, FILL[accent], rounded=True)
        self.box(slide, x, y+.2, .045, h-.4, accent)
        self.text(slide, x+.23, y+.16, w-.46, .42, title, title_size, accent, bold=True)
        self.text(slide, x+.23, y+.71, w-.46, h-.82, body, size)

    @staticmethod
    def notes(slide, text):
        slide.notes_slide.notes_text_frame.text = text.strip()


def composition(deck):
    """Connect registration and request-time selection through one registry."""
    s = deck.slide(1, "Dispatcher：从实现登记到选型与执行",
                   "先声明并登记实现，调用时匹配能力；kernel_id 将选择结果连接到同一条记录的 runner")
    # Registration flow, plus the registry's two distinct read relationships.
    deck.line(s, 4.60, 3.67, 5.00, 3.67, BLUE, arrow=True)
    deck.line(s, 10.70, 3.67, 11.10, 3.67, BLUE, arrow=True)
    deck.line(s, 11.60, 4.79, 11.60, 5.47, BLUE)
    deck.line(s, 11.60, 5.47, 6.85, 5.47, BLUE)
    deck.line(s, 6.85, 5.47, 6.85, 5.65, BLUE, arrow=True)
    deck.line(s, 14.60, 4.79, 14.60, 5.65, BLUE, arrow=True)
    for a, b in ((4.10, 4.50), (9.20, 9.60), (12.20, 12.60)):
        deck.line(s, a, 6.53, b, 6.53, BLUE, arrow=True)

    deck.text(s, .65, 2.15, 14.7, .35,
              "注册准备  ·  首次初始化时汇总实现，建立可选择的目录", 18, BLUE, bold=True)
    deck.box(s, .65, 2.62, 3.95, 2.17, PALE, BORDER, rounded=True)
    deck.text(s, .88, 2.80, 3.49, .41, "实现提供方", 21, BLUE, bold=True)
    for y, text, accent in (
        (3.38, "CANN：Python 声明 + API runner", TEAL),
        (3.72, "Triton：Python 声明 + JIT runner", PURPLE),
        (4.06, "Ascend C：manifest + 冻结 spec", AMBER),
    ):
        deck.text(s, .88, y, 3.49, .29, text, 14.5, accent)
    deck.text(s, .88, 4.48, 3.49, .27, "绑定执行入口，按需加载 / 编译", 13.5, MUTED)

    deck.box(s, 5.00, 2.62, 5.70, 2.17, FILL[BLUE], rounded=True)
    deck.text(s, 5.23, 2.80, 5.24, .41, "KernelDescriptor · 每个实现一条", 20, BLUE, bold=True)
    fields = (
        ("op / kernel_id", "语义归属 / 唯一实现身份"),
        ("backend / compile_mode", "来源分类 / 编译策略"),
        ("capability", "CapabilitySpec：支持域"),
        ("runner", "执行函数 / 可调用对象"),
    )
    for i, (name, role) in enumerate(fields):
        y = 3.32+i*.285
        deck.text(s, 5.23, y, 2.43, .27, name, 14.5, BLUE, bold=True)
        deck.text(s, 7.80, y, 2.67, .27, role, 14.5)
    deck.text(s, 5.23, 4.54, 5.24, .23,
              "辅助字段：priority / source / artifact / target_soc 等", 12, MUTED)

    deck.box(s, 11.10, 2.62, 4.25, 2.17, PALE, BORDER, rounded=True)
    deck.text(s, 11.33, 2.80, 3.79, .41, "KernelRegistry · 实现目录", 20, BLUE, bold=True)
    deck.text(s, 11.33, 3.38, 3.79, 1.06,
              "register()：按 kernel_id 唯一登记\n"
              "按 op 提供候选；按 ID 回查记录\n"
              "内置登记完成后冻结\n"
              "注册变更 → revision → 缓存失效", 14.5)

    deck.text(s, .65, 5.05, 10.55, .35,
              "运行调用  ·  以 Matmul 为例：调用事实 → 选型 → 决策 → 执行", 18, BLUE, bold=True)
    deck.text(s, 11.92, 5.02, 2.42, .44,
              "按 kernel_id\n回查同一目录", 12, MUTED)
    for x, w in ((.65, 3.45), (4.50, 4.70), (9.60, 2.60), (12.60, 2.75)):
        deck.box(s, x, 5.65, w, 2.18, FILL[BLUE], rounded=True)

    deck.text(s, .88, 5.82, 2.99, .40, "调用事实", 20, BLUE, bold=True)
    deck.text(s, .88, 6.29, 2.99, .32, "InputSignature", 17, BLUE, bold=True)
    deck.text(s, .88, 6.64, 2.99, .49,
              "op / dtype / shape / layout\n含预期 out 与特化参数", 13.5)
    deck.text(s, .88, 7.20, 2.99, .31, "HardwareSignature", 17, BLUE, bold=True)
    deck.text(s, .88, 7.54, 2.99, .26, "SoC / AI Core / UB", 13.5)

    deck.text(s, 4.73, 5.82, 4.24, .40, "Dispatcher · dispatch()", 20, BLUE, bold=True)
    deck.text(s, 4.73, 6.34, 4.24, 1.34,
              "先查决策缓存；未命中时：\n"
              "1. 只取同 op 的 Descriptor\n"
              "2. backend + 能力 / 硬件 / JIT 过滤\n"
              "3. 有效 tuning，否则确定性排序\n"
              "选来源栈再选栈内实现；空集报错", 14.5)

    deck.text(s, 9.78, 5.84, 2.24, .36, "DispatchDecision", 18, BLUE, bold=True)
    deck.text(s, 9.78, 6.34, 2.24, 1.32,
              "kernel_id / backend\nstack_id / reason\n候选与拒绝原因\ncache_hit\n输出选中身份与依据", 14)
    deck.text(s, 12.78, 5.82, 2.39, .40, "门面 → Runner", 19, BLUE, bold=True)
    deck.text(s, 12.78, 6.34, 2.39, 1.32,
              "公共层分配 / 复用 out\nRegistry.get(kernel_id)\n→ Descriptor.runner\nrunner(a,b,out)\n加载 / 编译 / host / launch", 13.5)

    deck.box(s, .65, 8.04, 14.7, .39, FILL[BLUE], rounded=True)
    deck.text(s, .86, 8.085, 14.28, .28,
              "选型读取 Descriptor 的能力声明，执行调用同一记录的 runner；kernel_id 连接两个阶段。", 15.5, BLUE)
    deck.notes(s, """
本页将注册准备与运行调用连接起来，两条从 KernelRegistry 向下的线表达不同的读取关系：
Dispatcher 按 op 取得候选描述记录；选型完成后门面按 decision.kernel_id 回查同一目录，
取得同一 KernelDescriptor 上绑定的 runner。不是为选择和执行分别维护两套注册表。

注册准备由实现提供方交付描述与执行入口：CANN 与 Triton 的 Python provider 返回
KernelDescriptor；Ascend C 从 registry.json 读取记录，关联冻结 spec/recipe，
bind_aot_kernel 绑定 AotKernelRunner。Triton 绑定 TritonMatmulRunner，CANN 绑定函数。
首次 get_global_registry() 汇总 _builtin_descriptors() 与 ascendc_descriptors()，
KernelRegistry 构造时逐条 register()，唯一 kernel_id 禁止覆盖，再 freeze 内置登记。
外部仓与用户扩展分别走 register_external_kernel/register_custom_kernel 的专用入口；
这类追加登记推进 revision，使下一次决策不复用旧目录状态的缓存。
注册阶段只准备元数据和 callable，真正加载产物或 JIT 编译在 runner 的执行阶段按需发生。

KernelDescriptor 是不可变实现描述，不是执行调度器。
op 表示语义接口，例如 gemm.matmul；kernel_id 唯一标识该接口的一个实现；
backend 是来源，compile_mode 是本仓编译策略，capability 是支持域，runner 是执行入口。
priority 用于确定性排序，source/artifact/target_soc 绑定源码、产物与目标身份，
图中只列核心字段，完整类型还包括 ABI、workspace、graph 声明等。
CANN 的 compile_mode=NONE 表示本仓不编译，厂商 kernel 仍是预编译的。
CapabilitySpec 使用 InputSignature + HardwareSignature 判断兼容性；compile_mode
同时参与 JIT 开关过滤与默认分档，真正的编译/加载行为由 runner 管理。

运行调用以当前 Matmul 为例，非所有门面分配时机的统一断言。
输入和预期 out 元数据组成 InputSignature，当前设备组成 HardwareSignature。
决策缓存未命中时，只取同 op 的记录；逐条执行 backend 策略、能力、target 与 JIT 过滤，
有效 tuning 冠军也必须通过这些过滤。无有效冠军时采用确定性排序：tier、priority、kernel_id，
先选来源栈再选栈内实现。Stack 是 Registry 派生视图，不是独立注册表。
CandidateReport/StackReport 记录逐项拒绝与排序原因；零候选抛 DispatchError。
复杂 shape×dtype 条件应继续扩展在兼容性过滤阶段，位于 tuning/排序之前。
相同决策 key 命中时可复用 DispatchDecision；key 包括签名、硬件、策略、revision 与 tuning token。
显式 kernel_id 是更高优先级的锁定分支，仍验 capability 与 target，本图展开 backend/auto 主路径。
DispatchDecision 只返回选中身份和依据；默认顺序不是实测性能最优声明。

执行：Matmul 公共门面在选型成功后分配/复用 out，并检查重叠，再调用 _run_selected。
公共 runtime.runner.run_selected（门面别名 _run_selected）做 Registry.get(id)，检查后调用 runner。
所以 get 与调用由门面执行，不是 runner 自己查找或 dispatcher 直接 launch kernel。
同一条 Descriptor 的能力用于选择、runner 用于执行，kernel_id 保持两者关联。
CANN runner 进入厂商预编译调用；AOT runner 加载 op_package；Triton runner 使用进程模块、
验证后的磁盘缓存，miss 才 JIT。决策缓存与二进制缓存的职责不同。
本例不自动遍历 fallback_chain 重试其他 runner。生产实现、精度证据与队列边界沿用后续页面。

源码依据：flashinfer/runtime/{schema,signature,hardware,registry,dispatch,runner}.py；
flashinfer/{gemm,ascendc}.py；flashinfer/cann/gemm/；flashinfer/triton/gemm/；
flashinfer/jit/runtime.py；build/generated/flashinfer/data/kernels/{registry,ascendc_specs}.json。
""")


def matmul(deck):
    """Trace a single out-schema invocation to the selected callable and kernel."""
    s = deck.slide(2, "一次 Matmul 调用：从统一接口逐层路由到 kernel",
                   '以 backend="ascendc" 为例展开；更换 backend 时沿用公共链路，再进入对应 runner')
    # One continuous call: first row left-to-right, then the second row.
    deck.line(s, 3.0, 2.68, 3.0, 2.92, BLUE, arrow=True)
    for y in (3.52, 5.02):
        for x1, x2 in ((5.35, 5.65), (10.35, 10.65)):
            deck.line(s, x1, y, x2, y, BLUE, arrow=True)
    deck.line(s, 13.0, 4.14, 13.0, 4.28, BLUE)
    deck.line(s, 13.0, 4.28, 3.0, 4.28, BLUE)
    deck.line(s, 3.0, 4.28, 3.0, 4.42, BLUE, arrow=True)
    deck.line(s, 13.0, 5.65, 13.0, 5.91, BLUE)
    deck.line(s, 3.0, 5.91, 13.0, 5.91, BLUE)
    for x, accent in ((3.0, BLUE), (8.0, AMBER), (13.0, BLUE)):
        deck.line(s, x, 5.91, x, 6.07, accent, width=1.9 if x == 8 else 1.5, arrow=True)

    deck.box(s, .65, 2.12, 14.7, .56, PALE, rounded=True)
    deck.text(s, .88, 2.25, 10.9, .35,
              'flashinfer.matmul(a, b, out=out, backend="ascendc")', 18, BLUE)
    deck.text(s, 12.02, 2.19, 3.10, .45,
              "BF16 / ND · Ascend950\nA[32,32] × B[32,64]", 12, MUTED)
    steps = (
        (.65, 2.92, 1.22, "① 公共 Torch 入口",
         "torch.ops.flashinfer.matmul_out\n→ _plain_matmul_out → _matmul_impl"),
        (5.65, 2.92, 1.22, "② 输入与硬件签名",
         "output_metadata → build_signature\n+ current_hardware()"),
        (10.65, 2.92, 1.22, "③ dispatch：取得候选",
         'get_global_registry()\n→ registry.stacks(op="gemm.matmul")'),
        (.65, 4.42, 1.23, "④ dispatch：过滤与选型",
         "backend / 能力 / 硬件 / JIT 过滤\n有效 tuning / 确定性规则选栈与实现"),
        (5.65, 4.42, 1.23, "⑤ DispatchDecision",
         "本例选中 AOT Matmul 的 kernel_id（Ka）\nstack_id / backend / reason / reports"),
        (10.65, 4.42, 1.23, "⑥ 公共输出与执行入口",
         "公共层准备 out；_run_selected(Ka,...)\n→ Registry.get(Ka).runner(a,b,out)"),
    )
    for x, y, h, title, body in steps:
        deck.box(s, x, y, 4.7, h, FILL[BLUE], rounded=True)
        deck.text(s, x+.20, y+.13, 4.30, .36, title, 18.5, BLUE, bold=True)
        deck.text(s, x+.20, y+.60, 4.30, .58, body, 14.5)

    branches = (
        (.65, TEAL, "cann · 厂商预编译", False,
         "_matmul_cann(a,b,out)\n→ torch.mm → TorchNPU / CANN\n→ 厂商预编译 Matmul kernel"),
        (5.65, AMBER, "ascendc · AOT（本例）", True,
         "AotKernelRunner.resolve → 加载 AOT 包\n→ TorchOpPackageRunner → 原生桥\n→ ACLNN host tiling / launch → kernel"),
        (10.65, PURPLE, "triton · JIT", False,
         "TritonMatmulRunner.prepare\n→ 模块 / 磁盘缓存；未命中时 JIT\n→ launcher → Triton kernel"),
    )
    for x, accent, title, selected, body in branches:
        deck.box(s, x, 6.07, 4.7, 1.73, FILL[accent], accent if selected else None, rounded=True)
        deck.box(s, x, 6.27, .045, 1.33, accent)
        deck.text(s, x+.23, 6.22, 4.24, .38, title, 19, accent, bold=True)
        deck.text(s, x+.23, 6.76, 4.24, .90, body, 14)
    deck.box(s, .65, 7.97, 14.7, .34, FILL[BLUE], rounded=True)
    deck.text(s, .86, 8.015, 14.28, .28,
              "一次调用只执行选中的 runner；公共接口与 Tensor 契约相同，后端差异在执行适配层展开。", 15, BLUE)
    deck.text(s, .65, 8.35, 14.7, .18,
              "图示展开决策缓存未命中路径；其余两分支表示更换 backend 后的路径。当前共同域：Ascend950 / 连续 ND BF16 / M、N、K 为 16…256 的 16 倍数。", 9.5, MUTED)
    deck.notes(s, """
本页沿单次具体调用解释公共入口、选型和执行。示例：
flashinfer.matmul(a,b,out=out,backend='ascendc')；a[32,32]、b[32,64]、out[32,64]，
Ascend950、连续 ND BF16。Ka 是当前 AOT Matmul 实现的唯一 kernel_id 的图中别名：
ascendc/gemm/matmul/native/op_package/Ascend950。
不传 kernel_id。框图中 CANN/Triton 是更换 backend 参数后的可选路径，本次只执行 AOT 分支。

① flashinfer.gemm.matmul 调用时自动注册公共 Torch schema/PrivateUse1 实现；
传入 out 时调用 torch.ops.flashinfer.matmul_out，经 _plain_matmul_out 进入 _matmul_impl。
函数与模块名中的 ACLNN/Ascend C 不代表已经选中了该后端；这里是三个后端共有的门面适配。
不传 out 则经 torch.ops.flashinfer.matmul 与 _plain_matmul 进入同一个 _matmul_impl。
② output_metadata 获取传入 out 或预期输出元数据，build_signature 构造 op=gemm.matmul 的
InputSignature，包含 dtype/shape/layout/physical_layout 与 tensor_contract；current_hardware
取得 HardwareSignature。out=None 时此阶段也不分配实际输出。
③ dispatch 取得同一全局 Registry，并先查决策缓存。图展开未命中路径：
registry.stacks(op=signature.op) 获取同 op 的描述记录及来源视图，其他 op 不参与选型。
首次访问 Registry 时，CANN/Triton Python providers 与 AOT manifest→Descriptor→runner 绑定的
结果统一登记，之后冻结。manifest 只是本仓 AOT 描述的当前来源，AOT 最终仍是普通 KernelDescriptor。
④ 对每条记录做 backend、CapabilitySpec、硬件目标与 JIT 开关过滤。显式 ascendc 排除其余来源，
仍须通过能力检查。有效 tuning 冠军必须在兼容者中；否则按 tier、priority、kernel_id 排序，
先选来源栈再选栈内实现。候选为空则 DispatchError；不按 provider 名称、登记顺序或函数名选型。
⑤ DispatchDecision 保存 Ka、backend=ascendc、stack_id=ascendc、reason、reports 等，
只返回选中身份与依据，不执行 kernel。未使用显式 kernel_id 的更高优先级锁定分支。
⑥ 当前调用复用传入 out，公共层拒绝输出与输入重叠；out=None 时在决策成功后统一分配。
门面的 _run_selected 是 runtime.runner.run_selected 的别名；它按 Ka 查记录，检查 runner 后调用。
框图 Registry.get(Ka).runner 是这段薄封装的等价关系；不是 Runner 自己选择后端。

AOT：Descriptor.runner 为 AotKernelRunner；__call__ 经 resolve 查找、校验并加载 op_package，
runner_factory→package_tensor_runner→TorchOpPackageRunner；后者验证 Tensor 并经 _native.prepare
绑定原生 TorchNPU 桥，调用 flashinfer_aclnn._execute→ACLNN GetWorkspaceSize/host tiling→launch→kernel。
已解析 module/inner runner 在进程内复用。Matmul recipe allow_jit_fallback=false，缺包明确失败。
CANN：把 backend 改为 cann 后同一选型链路选出 cann/gemm/matmul_v1，
Descriptor.runner=_matmul_cann，经 torch.mm(a,b,out=out) 调用厂商预编译实现；本仓不编译 CANN kernel。
Triton：把 backend 改为 triton 后选中 triton/gemm/matmul_v1/Ascend950，
TritonMatmulRunner.__call__→prepare→进程模块或经验证的磁盘缓存，miss 才 build_and_load 编译，
随后 launcher 写入相同 out。决策缓存和产物缓存是不同层次。
三条当前实现均为同步 eager，共同 BF16 契约，无重排、转换 dtype 或结果复制的分支。
没有现场性能竞赛、性能最优或 graph-capture 保证；队列分层和版本边界在后续 Launch 页说明。

源码依据：flashinfer/gemm.py；flashinfer/aclnn/_ops.py；flashinfer/runtime/{registry,dispatch,runner}.py；
flashinfer/_matmul.py；flashinfer/cann/gemm/；flashinfer/triton/gemm/；
flashinfer/jit/{runtime,tensor_adapter}.py；flashinfer/aclnn/_native.py；flashinfer/ascendc/__init__.py。
""")


def matmul_auto(deck, evidence):
    cases = {c["case"]: c for c in evidence["cases"]}
    assert evidence["kind"] == "HOST_ROUTING_ONLY_NO_KERNEL_EXECUTION"
    assert evidence["status"] == "PASS" and evidence["cases_passed"] == 12
    assert not evidence["global_registry_modified"]
    assert cases["omitted_backend_bf16"]["selected"] == "illustration/cann/bf16"
    assert cases["omitted_backend_fp16"]["selected"] == "illustration/triton/fp16"
    assert cases["omitted_obeys_global_cann"]["outcome"] == "EXPECTED_REJECTION"
    s = deck.slide(3, "Matmul 场景二：缺省 backend，按能力自动选择",
                   "由 CapabilitySpec 匹配 InputSignature；门面无需写 dtype → backend 分支")
    deck.box(s, .65, 2.17, 14.7, .56, PALE, rounded=True)
    deck.text(s, .88, 2.28, 7.8, .35, "flashinfer.matmul(a, b, out=out)", 20, BLUE)
    deck.text(s, 8.75, 2.31, 6.37, .33, "前提：backend=None，且全局默认 auto", 16, MUTED)
    deck.line(s, 4.85, 4.35, 5.25, 4.35, BLUE, arrow=True)
    deck.box(s, .65, 3.02, 4.20, 2.76, PALE, BORDER, rounded=True)
    deck.text(s, .88, 3.18, 3.74, .4, "KernelRegistry · 假设配置", 19, BLUE, bold=True)
    deck.text(s, .88, 3.66, 3.74, .3, "同一 op、同一 shape / layout / SoC", 14.5, MUTED)
    for y, accent, title, dtype in ((4.10, TEAL, "CANN descriptor · Kc", "bfloat16"),
                                   (4.90, PURPLE, "Triton descriptor · Kt", "float16")):
        deck.box(s, .88, y, 3.74, .65, FILL[accent], rounded=True)
        deck.text(s, 1.04, y+.06, 3.42, .29, title, 16, accent, bold=True)
        deck.text(s, 1.04, y+.35, 3.42, .26, "支持 dtype："+dtype, 13.5)

    deck.text(s, 5.25, 3.10, 10.1, .39, "Dispatcher：逐条检查能力，生成 CandidateReport", 18, BLUE, bold=True)
    tx = (5.25, 8.12, 10.51, 12.90)
    widths = (2.75, 2.27, 2.27, 2.45)
    for x, cw, label, accent in zip(tx, widths,
                                    ("InputSignature", "CANN · Kc", "Triton · Kt", "DispatchDecision"),
                                    (BLUE, TEAL, PURPLE, BLUE)):
        deck.box(s, x, 3.67, cw, .40, FILL[accent], rounded=True)
        deck.text(s, x+.13, 3.72, cw-.26, .29, label, 14.5, accent, bold=True)
    for y, dtype, selected in ((4.25, "bfloat16", "Kc"), (4.93, "float16", "Kt")):
        for x, cw in zip(tx, widths):
            deck.box(s, x, y, cw, .50, PALE, rounded=True)
        deck.text(s, tx[0]+.13, y+.10, widths[0]-.26, .31, "dtype="+dtype, 16)
        for column, alias, accent in ((1, "Kc", TEAL), (2, "Kt", PURPLE)):
            passed = selected == alias
            deck.text(s, tx[column]+.13, y+.10, widths[column]-.26, .31,
                      "通过" if passed else "dtype 不匹配", 15.5, accent if passed else MUTED)
        deck.text(s, tx[3]+.13, y+.10, widths[3]-.26, .31,
                  "kernel_id="+selected, 15.5, TEAL if selected == "Kc" else PURPLE, bold=True)
    deck.text(s, 5.25, 5.51, 10.1, .29,
              "示例 shape：A[32,32] × B[32,64] → out[32,64]；HardwareSignature = Ascend950", 13, MUTED)
    deck.line(s, 14.125, 5.43, 14.125, 5.99, BLUE, arrow=True)
    deck.box(s, .65, 5.99, 14.7, .87, FILL[BLUE], rounded=True)
    deck.text(s, .88, 6.10, 14.24, .60,
              "决策输出 Kc / Kt → Registry.get(kernel_id) → KernelDescriptor.runner(a,b,out)\n"
              "执行仍走同一链路：Kc → CANN 预编译 kernel；Kt → Triton JIT / 缓存 → kernel", 16.5, BLUE)
    for x, title in ((.65, "1 个候选：该 kernel 成为冠军"), (5.65, "多个：tuning / 确定性排序"),
                     (10.65, "0 个：DispatchError + 拒绝原因")):
        deck.box(s, x, 7.10, 4.7, .52, PALE, rounded=True)
        deck.text(s, x+.20, 7.20, 4.3, .32, title, 15.5, MUTED)
    deck.box(s, .65, 7.92, 14.7, .43, FILL[AMBER], rounded=True)
    deck.text(s, .87, 7.985, 14.26, .30,
              "能力机制示例：门面与 runner 也须允许 FP16；当前生产三后端 Matmul 的共同支持域仍为 BF16。", 14.5, AMBER)
    deck.text(s, .65, 8.36, 14.7, .19,
              "省略 backend 读取全局配置（缺省 auto）；显式 backend='auto' 可覆盖全局默认。本页只验证元数据选择，不执行 FP16 kernel。", 10.1, MUTED)
    deck.notes(s, """
本页是用户提出的能力配置示例，不是生产 Matmul FP16 支持声明。
假设同一语义 op 有两个实现：CANN 只接受 BF16，Triton 只接受 FP16；
shape/layout/SoC 等其他约束均满足，JIT 开启，且公共门面和各 runner 契约也已允许相应 dtype。
这两条记录分别以 Kc/Kt 指代，唯一 kernel_id 与各自 runner 绑定；不与其他 op 的描述记录混选。

调用 flashinfer.matmul(a,b,out=out) 省略 backend 时传 backend=None，
dispatch 读取 FLASHINFER_NPU_BACKEND，未设置时默认 auto。若全局强制 cann，则 FP16 仍会失败；
想绕过全局默认可显式 backend='auto'。auto 并不是无条件由省略参数触发。
InputSignature 保存真实 dtype、shape、预期 out 等；HardwareSignature 保存硬件事实。
KernelRegistry 提供实现记录，Dispatcher 使用每条 descriptor.capability 的共享兼容性谓词，
检查 dtype/shape/layout/SoC 等及策略。不是在 Matmul 门面写 if dtype: backend=...。
BF16 输入时 CANN 通过、Triton 因 dtype 拒绝；FP16 输入时结论相反，CandidateReport 保存逐条原因。
本例每次只有一个通过者，因此该实现成为冠军；仍经既有 tuning 查询和来源栈/栈内选择路径。
多个通过者采用有效 tuning 冠军，否则按 tier、priority、kernel_id 确定性排序；
零通过者抛 DispatchError。tuning 不会复活已经被 dtype、shape、硬件或策略剔除的实现。
此处没有现场 kernel 性能测量，也不把确定性顺序称为实测最优。
Decision 输出 kernel_id 后，统一分配/复用 out，再 get(id).runner(a,b,out)，进入对应的执行适配。
dtype/shape 是决策缓存 key 的组成部分，不会把 BF16 的缓存决策用于 FP16。
未来 shape×dtype 级联能力条件扩展在同一兼容性过滤阶段，位于 tuning/排序之前。

实际元数据验证：tools/routing_capability_demo.py 本轮运行 12 项均通过。
该工具使用隔离 KernelRegistry、示例 op 和元数据签名，不改全局登记，不分配 Tensor，不执行 kernel。
覆盖两种缺省路由、强制后端不兼容、全局 backend 覆盖、显式 auto、JIT 关闭、
不支持 dtype 以及 shape×dtype 配置组合。图中保留相同真实 dispatcher 组件与判断顺序。
当前生产 gemm.matmul 的三后端共同域是 ND BF16；tensor_contract_specialization、
公共 recipe 和各 runner 也限制此域，只把 Triton descriptor 改成 FP16 并不能让生产 API 支持 FP16。
本次只扩展讲解与元数据验证，不新增生产 FP16 kernel 或放宽公共契约。

源码依据：flashinfer/gemm.py；flashinfer/config.py；flashinfer/runtime/{signature,schema,dispatch,registry}.py；
flashinfer/_matmul.py；tools/routing_capability_demo.py。
""")


def registration(deck):
    s = deck.slide(4, "后端接入：统一登记契约，分离构建与 Launch",
                   "每个实现都要登记；Runner 屏蔽编译、产物加载与底层提交方式的差异")
    deck.box(s, .65, 2.20, 14.7, .56, FILL[BLUE], rounded=True)
    deck.text(s, .88, 2.30, 14.24, .35,
              "统一记录：op + 唯一 kernel_id + backend + CapabilitySpec + runner → KernelRegistry", 17, BLUE)
    xs, cw = (2.03, 6.51, 10.99), 4.36
    headers = ((TEAL, "CANN · 厂商预编译"), (AMBER, "Ascend C · AOT"), (PURPLE, "Triton · JIT"))
    for x, (accent, title) in zip(xs, headers):
        deck.box(s, x, 3.03, cw, .56, accent, rounded=True)
        deck.text(s, x+.16, 3.125, cw-.32, .35, title, 19, WHITE, bold=True)
    rows = (
        (3.73, 1.04, "登记", (
            "Python descriptor\n绑定既有 API runner",
            "manifest + 冻结 spec / recipe\n绑定 AOT runner",
            "Python descriptor + SourceRef\n绑定 JIT runner")),
        (4.92, .96, "交付", (
            "依赖 CANN / TorchNPU\n使用其预编译包",
            "wheel 随包提供\nop_package + 原生 TorchNPU 桥",
            "kernel 源码 + JIT 工具链\n运行时管理编译缓存")),
        (6.03, 1.37, "Launch", (
            "torch.mm → TorchNPU\n→ CANN 内置 host / kernel",
            "原生桥 → RunOpApi\n→ ACLNN（host tiling / launch）\n→ native kernel",
            "JIT / 缓存模块 → launcher\n→ OpCommand custom handler\n→ runtime → Triton kernel")),
    )
    for y, h, title, values in rows:
        deck.text(s, .67, y+.16, 1.20, .40, title, 18.5, MUTED, bold=True)
        for x, (accent, _), body in zip(xs, headers, values):
            deck.box(s, x, y, cw, h, FILL[accent], rounded=True)
            deck.text(s, x+.17, y+.15, cw-.34, h-.22, body, 16.5)
    deck.box(s, .65, 7.66, 14.7, .44, FILL[BLUE], rounded=True)
    deck.text(s, .87, 7.725, 14.26, .31,
              "执行汇合点：相同 Tensor 契约、逻辑 NPU device 与当前 stream。", 16, BLUE)
    deck.text(s, .65, 8.17, 14.7, .32,
              "队列开启时，AOT 与本机 Triton 均可经 TorchNPU 入队；共享 stream ≠ 相同 RTS 入口或物理队列。", 13.1, MUTED)
    deck.notes(s, """
登记的共性：每个实现独立提供 KernelDescriptor，属于同一语义 op gemm.matmul，
kernel_id 唯一，声明 backend/CapabilitySpec/compile_mode 并绑定 runner；同一 Registry 统一选择。
backend 是来源分类，compile_mode 是本仓的编译策略，两者不能混为一谈。
CANN 在 cann.gemm.gemm_descriptors 中构造 Python descriptor，cann_descriptors 汇总。
Ascend C AOT 的 registry.json 提供 descriptor，ascendc_specs.json 提供冻结 spec，
build/generated/csrc/recipes/matmul.json 描述 host/kernel 源码与契约；ascendc_descriptors 经 bind_aot_kernel
绑定 AotKernelRunner。交付 op_package、host tiling/API、device binary、manifest 与原生桥。
Triton 在 triton.gemm.gemm_descriptors 提供 descriptor + SourceRef，绑定延迟构建的
TritonMatmulRunner；triton_descriptors 汇总。运行时生成 JitSpec，统一 build/load 并复用产物。
各 provider 最终都汇入全局 KernelRegistry。多个 gemm_descriptors 是 provider 函数，不是独立路由器。
第三方仓已有 TorchNPU API 可以包装成 runner，但必须满足相同输出、layout、device 和 stream 契约；
经验证后以 register_external 和仓库自己的命名空间登记，不冒充 CANN。

Launch 和队列须分层：Python runner / host tiling → TorchNPU host task_queue（可选）→
runtime stream → device kernel；host 队列不是 device 硬件 SQ/CQ。
Native AOT 桥调用 OpCommand::RunOpApi(name,callback,true)，普通 launch stream 且队列开启时
可提交 EXECUTE_OPAPI；sync=true 在其后等待 stream，并不阻止前面的入队。
本机 Triton 3.6.0 的实际缓存 launcher 通过 g_async_launch/triton_async_launch，
在 OpCommand::Run custom handler 的对应条件下提交 COMPILE_AND_EXECUTE；
这个任务枚举名不代表每次重新 JIT。生成策略受 TRITON_ENABLE_TASKQUEUE 影响，须以缓存产物为准。
真实缓存的 npu_utils 引用 Run/SetCustomHandler，launcher 导入 aclrtLaunchKernelWithHostArgs。
CANN 由 torch.mm 委托 TorchNPU/CANN，本材料未追踪其准确 task 类型和最底层 RTS 入口。
因此只能说本例共享调用方当前逻辑 stream，不能说三者使用完全相同 RTS 函数或物理队列。
上述 queue 解释限定本机 TorchNPU 2.10.0.post7 与 Triton 3.6.0；未采集逐 task profiler trace。

源码与证据：flashinfer/runtime/registry.py；
flashinfer/cann/gemm/；flashinfer/ascendc/__init__.py；flashinfer/triton/gemm/；
build/generated/flashinfer/data/kernels/{registry,ascendc_specs}.json；build/generated/csrc/recipes/matmul.json；
docs/matmul-backends.md；docs/aclnn-torch-adapter.md；
docs/flash-attention-call-chain.md。
固定上游：Ascend/pytorch commit 3363ea59d49f3e08cfd8d4a1f532b756ccda0682，
torch_npu/csrc/framework/OpCommand.cpp（与本机 git_version 一致）。
""")


def validate(deck, destination, font_path):
    """Check the deliverable itself, plus text widths using the selected face."""
    assert len(deck.prs.slides) == deck.total
    renderer = FigureCanvasAgg(Figure(figsize=(WIDTH, HEIGHT), dpi=72)).get_renderer()
    warnings = []
    for item in deck.texts:
        prop = FontProperties(fname=str(font_path), size=item["size"])
        for line in item["content"].splitlines():
            width, _, _ = renderer.get_text_width_height_descent(line, prop, False)
            # Reserve a little extra space for a different office text renderer.
            if width > item["w"]*72-2:
                warnings.append(f"Slide {item['slide']}: text width {width:.1f}pt / "
                                f"{item['w']*72:.1f}pt: {line}")
        needed = len(item["content"].splitlines()) * item["size"] * 1.27
        if needed > item["h"]*72+1:
            warnings.append(f"Slide {item['slide']}: text height {needed:.1f}pt / "
                            f"{item['h']*72:.1f}pt: {item['content']}")
    for index, slide in enumerate(deck.prs.slides, 1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (index, shape.name)
            assert shape.left+shape.width <= deck.prs.slide_width+2, (index, shape.name)
            assert shape.top+shape.height <= deck.prs.slide_height+2, (index, shape.name)
    assert not warnings, "\n".join(warnings)
    with ZipFile(destination) as z:
        slides = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
        notes = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n))
        assert len(slides) == len(notes) == deck.total
        namespaces = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        counts = []
        for name in slides:
            xml = etree.fromstring(z.read(name))
            assert not xml.findall(".//p:pic", namespaces), "Slides must remain editable"
            visible = "\n".join(xml.xpath(".//a:t/text()", namespaces=namespaces))
            assert not re.search(r"\.py:\d+|/mnt/fi/|§\d", visible)
            counts.append(dict(shapes=len(xml.findall(".//p:sp", namespaces)),
                               connectors=len(xml.findall(".//p:cxnSp", namespaces)),
                               text_runs=len(xml.findall(".//a:t", namespaces))))
    return {"slides": deck.total, "native_editable": True, "slide_objects": counts,
            "speaker_notes": deck.total, "text_boxes": len(deck.texts),
            "bounds_and_text_metrics": "PASS", "no_source_lines_on_slides": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/presentations")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    args = parser.parse_args()
    assert args.font.is_file(), f"Chinese font is required: {args.font}"
    # This demo uses isolated registries and metadata only; it never calls a runner.
    sys.path.insert(0, str(ROOT))
    from routing_capability_demo import run as check_routing

    evidence = check_routing()
    deck = Deck()
    composition(deck)
    matmul(deck)
    matmul_auto(deck, evidence)
    registration(deck)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (STEM+".pptx")
    deck.prs.save(str(output))
    result = validate(deck, output, args.font)
    result["pptx"] = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
    result["pptx_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    result["sources"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                         for name in SOURCE_PATHS}
    result["routing_demo"] = {"cases_passed": evidence["cases_passed"], "kind": evidence["kind"]}
    for name, payload in (("validation", result), ("slide-text", deck.texts), ("capability-demo", evidence)):
        (args.artifacts_dir/(name+".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k,v in result.items() if k != "sources"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
