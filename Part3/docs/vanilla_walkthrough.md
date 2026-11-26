# 从真实 Part1 数据看懂 vanilla 七阶段

这份说明对应当前 `Part3` 的代码和 `cases/team_handoff`，先看数据怎样算，再看硬件怎样搬。
这里运行的是 VGG16 中的一个卷积层，不是把整个 VGG16 放进 FPGA。

## 1. 先认识这一小层的输入和权重

Part1 选中 `features.27`：8 输入通道、8 输出通道、3×3 kernel、stride=1、padding=1、没有 bias。
层输入是 `1×8×4×4`：一张样本、八个通道，每个通道 4×4。
权重是 `8×8×3×3`；第一个 8 是输出通道，第二个 8 是输入通道。
输出仍是 `1×8×4×4`，共 `8×4×4=128` 个整数。
后面只有 ReLU，没有 BN；模型中的下一层卷积是 `features.29`。

| 当前 case 文件 | 怎样理解 |
|---|---|
| `input.txt` | 已经补零成 8×6×6；36 行，每行打包八个通道，共 32 bit |
| `weight.txt` | 九个 kernel 位置，每个位置八行输出通道权重，共 72 行、每行 32 bit |
| `psum_pre_relu.txt` | 16 行、每行八个 16 bit 数：累加完、尚未 ReLU |
| `psum.txt` | 相同位置的 ReLU 后结果，是最终硬件 golden |

每个 word 的 channel 0 在最低位；文本从左到右显示 channel 7 到 channel 0。
activation 是 unsigned 4 bit，真实样本范围 0…11；weight 是 signed 4 bit，范围 −7…7。
例如 activation 的 `1011` 是 11，weight 的 `1110` 是 −2；乘法必须得到 −22。
[mac.v:13](../reference_dut/mac.v) 用零扩展保留 unsigned activation，再与 signed weight 相乘。

原始 Part1 文件带 header；[import_handoff.py:77](../import_handoff.py) 检查来源、header、alias 和两份 golden 后，生成这里的无 header case。
不要再次 padding，也不要把二进制文本当成十进制数字。

## 2. 一个输出为什么有 72 个乘积

设补零后的输入为 `A_pad[ic,y,x]`，一个输出通道、一个输出位置的数学定义是：

```text
Z[oc,oy,ox] = Σky=0..2 Σkx=0..2 Σic=0..7
               A_pad[ic,oy+ky,ox+kx] × W[oc,ic,ky,kx]
Y[oc,oy,ox] = max(0, Z[oc,oy,ox])
```

因此一个输出有 `3×3×8=72` 个乘积；整个 128 个输出共有 9216 个数学乘积项，包括值为零的项。
这里的“卷积”使用 PyTorch Conv2d 的 cross-correlation 顺序，不翻转 kernel。
[run.py:72](../run.py) 的 `integer_reference()` 用 Python 整数重算同一件事，并检查 16 bit 是否溢出。

8×8 阵列一次驻留的只是某个 `(ky,kx)` 的 `8×8` 权重矩阵，不是全部 576 个权重。
把一个输入位置的八通道向量送入后，八列产生八个输出通道的 dot product；每个 dot product只包含八个乘积。
行之间向下传递并累加 psum，activation 向东传播，权重暂时留在 PE 内，这就是此处的 weight stationary。
还要换九组 kernel 权重，再把九份贡献相加，才得到完整的 72 项结果。
阵列有流水延迟，送入一个向量也不等于同一时钟立刻拿到完整输出。

## 3. core、corelet、TB 分别是什么

```text
core_tb：用 task 发控制、提供输入、读 SRAM、核对结果
└─ core：把 corelet 和两个 SRAM 接起来
   ├─ 32 bit input SRAM：保存 activation 和 weight
   ├─ corelet：L0 → MAC array → OFIFO；另有八路 SFU
   └─ 128 bit psum SRAM：保存 kernel psum、完整和、ReLU 结果
```

[core.v:50](../reference_dut/core.v) 连接这些模块；[corelet.v:42](../reference_dut/corelet.v) 选择写入 SRAM 的数据来自 OFIFO 还是 SFU。
TB 是仿真控制器；`reference_dut` 是它驱动的硬件。当前 TB 没有把控制 task 综合成板上的硬件控制器。
基线通过 L0 从西边加载权重，不使用 IFIFO。

## 4. 运行入口怎样接上真实数据

在仓库的 `Part3/` 目录执行：

```powershell
python import_handoff.py
python run.py --sim vivado --case cases/team_handoff --repeat 2 --waves
```

[run.py:104](../run.py) 的 `prepare()` 读取 case，先核对给定 golden，再生成 `.mem` 和 `case_config.vh`。
TB 在 [core_tb.v:264](../tb/core_tb.v) 读这些文件，随后依次调用七个阶段；不是直接把期望值写到结果 SRAM。

| 阶段 | 当前 task 和代码位置 | 实际做什么 |
|---|---|---|
| 1 | `load_scratchpad()`，TB:143 | 把 36 个 activation word、72 个 weight word 写进 input SRAM |
| 2 | `load_pe_kernel(k)`，TB:153 | 复位计算状态，经 L0 给 PE 装本轮八个输出通道的权重向量 |
| 3 | `load_l0_activations(k)`，TB:167 | 从 input SRAM 把 36 个 padded 位置写进 L0 |
| 4 | `execute_kernel(k)`，TB:175 | 送入 36 个位置，等待流水排空 |
| 5 | `store_partials(k)`，TB:186 | 从 OFIFO 取八通道 psum，写入 psum SRAM |
| 6 | `accumulate_outputs()`，TB:221 | 为每个有效输出位置 gather 九个 word，经 SFU 累加并回写 SRAM |
| 7 | `relu_outputs()`，TB:245 | 读完整和，做 ReLU，再写 SRAM并读回核对 |

完整调用顺序见 [core_tb.v:278](../tb/core_tb.v)：阶段 2…5 对九个 kernel 位置各跑一次，之后才是阶段 6、7。
阶段 50 是额外的 raw psum 读回检查，不是课程又多出一个功能阶段；周期单独记在 `cycles.csv`。

## 5. SRAM 时序为什么需要两个上升沿

[sram_32b_w2048.v:16](../reference_dut/sram_32b_w2048.v) 的读地址在上升沿进入 `add_q`，随后 `Q=memory[add_q]` 才更新。
若同一个上升沿让 L0 捕获 Q，它看到的是更新前的数据，可能把上一地址的数据写进去。
因此 `xmem_to_l0()`（[TB:114](../tb/core_tb.v)）这样做：

1. 先给地址、`CEN=0,WEN=1`，经过第一个上升沿，让 SRAM 输出该地址的数据。
2. 再给 `l0_wr=1`，经过第二个上升沿，让 L0 捕获已经稳定的数据。

`CEN=0,WEN=0` 是写；`CEN=1` 是停用。这些控制都是低有效，不能看名字就当 1 表示开启。
TB 在下降沿改变控制，`tick()` 等待下一个上升沿完成采样再返回，避免 TB 与寄存器在同一沿抢先后。
OFIFO 也有读请求延迟；`store_partials()` 先让 SRAM 保存当前 head，再等 FIFO 消费它，不能连续错位读。

## 6. 为什么存了 324 个 psum，却只输出 16 个 word

这一基线每组 kernel 权重都计算全部 36 个 padded 位置，因此保存 `9×36=324` 个八通道 psum word。
设 `k=3*ky+kx`，每轮的存储地址是 `k*36+pos`。
完整输出的 16 个位置只需要各轮中相应移位后的输入位置，不能把同一 `pos` 的九个 word直接相加。

对输出 `p=oy*4+ox`，阶段 6 的地址公式为：

```text
input_pos = (oy+ky)*6 + ox+kx
psum_addr = k*36 + input_pos
```

例如左上角输出 `p=0`，必须区分输入位置与 psum SRAM 地址：

| kernel k | 本轮输入位置 | gather 的 SRAM 地址 |
|---|---|---|
| 0,1,2 | 0,1,2 | 0,37,74 |
| 3,4,5 | 6,7,8 | 114,151,188 |
| 6,7,8 | 12,13,14 | 228,265,302 |

所以完整 gather 列表是 `[0,37,74,114,151,188,228,265,302]`，九个 word各包含八个输出通道。
这与 Part1 的 `acc_address.txt` 一致；导入器逐项验证 144 个地址，TB 则从几何参数直接算地址。
当前 raw 区为 0…323，完整和区为 324…339，ReLU 区为 340…355；三者不互相覆盖。

## 7. SFU 必须先把九项加完，再做 ReLU

[sfp.v:18](../reference_dut/sfp.v) 的控制很直接：`clear` 清零；`acc` 加输入；`relu` 把负数变零；都为 0 时保持原值。
阶段 6 为每个输出位置先 clear，再读一个 SRAM word、发一次 acc，重复九次；读 SRAM 的等待期间必须 hold。
八个 SFU 并行处理 word 中的八个输出通道。加完后，`write_sfp_result()` 真正写回完整和区，再读回检查。
阶段 7 从完整和区读数，通过 `0+sum` 装入 SFU，发 ReLU，然后写回 ReLU 区；不是只看一眼 `sfp_out`。

看真实交接数据的一个数：输出位置 `p=1`，即 `(oy=0,ox=1)`，输出通道 `oc=7`。
九个 kernel 位置的八通道 dot product依次为：

```text
[0, 0, 0, -18, 14, 7, 2, 5, -20]
总和 Z = -10；最终 ReLU(Z) = 0
```

其中 `k=3` 的 activation 向量是 `[0,0,0,0,0,4,2,0]`，weight 向量是 `[6,0,-1,2,2,-2,-5,-3]`。
该项就是 `4×(-2)+2×(-5)=-18`；这是八项 dot product的一部分，不是完整卷积输出。
若每次更新 running sum 后就做 ReLU，会错误得到 8；若先对九个 dot product分别 ReLU再相加，则错误得到 28。
两个错误都会丢掉应当参与后续抵消的负数，正确结果只能是全部加完后的 0。

## 8. PASS 证明了什么，去哪里看

这次 Vivado 2017.4 的两次任务都通过：每次最终 SRAM 16 个 word、128 个标量；合计 256 个最终值精确匹配。
另外检查每次 324 个 raw psum word，以及完整和、ReLU 的 SRAM 回写；日志在 `build/team_handoff/simulation.log`。
`result_0.txt`、`result_1.txt` 是实际从最终 SRAM 读出的结果，`core_tb.vcd` 是控制与 SRAM 接口波形。
TB:127 的 `check_word()` 用 `!==` 检查，数值不符或出现 X/Z 都会失败；`run.py` 还会重读结果文件再比较。

不要把这件事与模型准确率混在一起：90.94% 是 Part1 原训练记录的 9094/10000 分类结果，这里没有重新测分类准确率。
模型侧最大误差 `4.076379191531032e-6`，是整数结果乘 activation/weight scale 后与真实下一层 prehook 的浮点值比较。
硬件 golden仍在整数域，必须零 mismatch，不能把 `1e-5` 浮点容差拿来放过整数计算错误。

## 9. 代码来源与当前边界

`reference_dut` 来自 2024 的 Part3alpha；原件未改，复制件的 SFU控制和回写扩展记录在 `docs/reference_hashes.json`、`docs/interface.md`。
官方 2026 starter 已原样归档到 `docs/starter2026/`，固定 main 为 `8f7df9b3e73b65d373c5e4bd738fd7162f41d046`。
官方 34 bit `inst` 字段与当前 TB 一致，但模板有五处 `...`；当前 task补足这些阶段，并加显式 SFU控制和真实 SRAM 回写检查。
核对结果见 [compatibility.json](starter2026/compatibility.json)，真实输入来源和哈希可在导入后生成的 `cases/team_handoff/provenance.json` 查阅。
这次通过的是参考硬件加 TB 的单层仿真；没有证明完整 VGG 硬件推理，也不是 Quartus/FPGA 的频率或功耗结果。
