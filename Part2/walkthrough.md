# 视频部分：从两个乘法例子读懂整帧控制

普通单层的七阶段见 [Step 3](../Step3/HANDOFF.md)。这里解释视频部分。

## 1. 我们现在到底在算什么

老师新版 README 明确：每个帧文件有 2048 个十进制数，shape 是 `[8,16,16]`，CHW 顺序，数值是 unsigned 8-bit。连续 256 行属于一个通道，再进入下一个通道。

四帧共用同一组 W。本实验采用同学训练后的 `features.27` 权重：8 个输入通道、8 个输出通道、3×3、stride 1、padding 1、无 bias。这是配置中写明的实验选择；老师四帧没有另外附 W。

每帧输出 `8×16×16=2048` 个数，四帧共 8192 个。这里算的是原始卷积部分和，包含负数。没有把四帧送进完整 VGG 做分类，也没有把视频整数结果直接乘上 Part1 的 4-bit 模型 scale。

老师 11 月 23 日换成 unsigned 数据。新版四帧放在仓库根目录的 `differential_input_batch_updated/`。

## 2. 为什么第一帧要分两次，后面每帧只需一次

PE 一次接收一个 4-bit activation。完整 8-bit 数可以精确写成：

```text
X = 16 × high + low
high = X >> 4       范围 0…15
low  = X & 15       范围 0…15
```

例如新版 X0 的第一项是 66，拆开是 high=4、low=2。暂时只看一个权重为 2 的乘法：

```text
先算 2×4 = 8
再算 2×2 = 4
合并 16×8 + 4 = 132 = 2×66
```

这只是一个乘法例子，完整输出还要对所有通道和 kernel 位置求和。

X1 同一项是 64，差分 `64−66=−2`。同一个乘法只需再算 `2×(−2)=−4`，然后 `132−4=128=2×64`。卷积是加法与乘法的组合，同样满足：

```text
Y0 = 16×Conv(X0_high,W) + Conv(X0_low,W)
Di = Xi − X0
Yi = Y0 + Conv(Di,W)
```

三份差分都对 X0 求，不是对上一帧求。新版实际差分均为 −7…7，没有截断或饱和。因此这里要求逐元素零误差。若未来输入超出 −8…7，程序会报错，不会偷偷截低四位。

普通方法的顺序：`X0_high, X0_low, X1_high, X1_low, X2_high, X2_low, X3_high, X3_low`。
差分方法的顺序：`X0_high, X0_low, D1, D2, D3`。

前者八个 pass，后者五个。省掉的是三个 8-bit 帧各自的一次额外 nibble 计算。

## 3. Python 准备数据，RTL 真正执行与重建

对应 `Part2/run_video.py`：

| 函数 | 解决的问题 |
|---|---|
| `load_case()` | 按配置读四帧与 W，检查尺寸、unsigned8 范围、signed4 差分范围，记录来源 hash |
| `passes()` | 准备八份普通输入、五份差分输入；给 delta 标记 signed |
| `spatial_words()` | 把 CHW 文件转为“一个位置的八通道一字”；channel 0 放最低四位 |
| `convolution()` | 直接做完整整数卷积，产生与 RTL 调度独立的参考 |
| `kernel_partials()` | 产生每个 kernel 位置的参考，方便定位通道、符号或地址错误 |
| `prepare()` | 写 `.mem` 和配置，先检查分解公式，再交给仿真 |
| `simulate()` | 编译、运行，读取实际 SRAM 导出的八个结果文件并逐帧比较 |
| `read_metrics()` | 核对每次权重加载和 pass 数，汇总实测周期 |

Python 的 `reconstruct()` 用于产生、检查参考答案。硬件输出没有交给 Python 相加冒充重建；TB 通过 SFU 控制完成移位、相加和 SRAM 写回，Python 最后只读结果比较。

差分在 Python 中生成，对应项目允许的 host/DRAM 处理方式。真实四帧需要 6144 次标量减法；这项主机开销没有计入 RTL 周期，报告必须保留这一点。

## 4. 一个 kernel 位置怎样覆盖整帧

8×8 阵列同时保存一个 `(ky,kx)` 位置的 64 个权重：八输入通道乘八输出通道。完整 3×3 层有九组这样的权重。

L0 和 OFIFO 各只能容纳 64 个位置，整帧却有 256 个输出位置。因此控制顺序为：

```text
for kernel_index = 0…8:
    加载这一个 kernel 位置的 W，只加载一次
    for pass_index = 0…4:           普通方法这里是 0…7
        把这一份 activation 从模拟 DRAM 写到 L1
        for chunk_start = 0,64,128,192:
            清数据流水，保留 W
            根据输出坐标，从 L1 取对应输入，装入 L0
            运行 PE，排空流水
            经 OFIFO 把结果写进临时 psum SRAM
        用 SFU 累加到这一 pass 的整层部分和
```

先把同一 W 的五个 pass 全部做完，再换下一个 kernel 位置。这满足跨帧复用 W 的要求。若先算完整 X0 再回头处理 D1，会重新装九组 W，失去这里的复用。

真实配置下：差分是 `9×5=45` 个 kernel-pass、`45×4=180` 次 chunk 执行。普通方法是 72 个 kernel-pass、288 次 chunk 执行。逻辑上 8→5 的比较和实际阵列执行次数要同时讲清楚。

## 5. 对应到你的 testbench task

打开 `Part2/tb/video_tb.v`，按下面的顺序读：

| task | 实际做的事 | 主要控制 |
|---|---|---|
| `load_kernel()` | 全局 reset 开启新 W 会话，经 L0 装八列权重；检查每个 PE 的 W 是否与源文件一致 | `load`、`l0_rd` |
| `stage_activations()` | 把当前 pass 的 256 个 packed word 写到 L1，另留一个零 word | SRAM 的 `CEN/WEN/A/D` |
| `clear_pass()` | 清上一次 chunk 的流水、FIFO 指针和 SFU，保留权重 | `compute_clear` |
| `gather_chunk()` | 从输出位置反推当前 kernel 所需的输入位置；仅图像外边界取零 | `xmem_word_to_l0()` |
| `execute_chunk()` | 从 L0 读 activation，流水运行并等待最后数据离开阵列 | `execute`、`l0_rd` |
| `store_chunk()` | 等 OFIFO 有完整八列结果，写到临时 psum SRAM | `ofifo_valid`、`ofifo_rd` |
| `accumulate_pass()` | 通过 SFU 累加九个 kernel 位置；普通方法还按 high/low 做移位 | `sfp_clear`、`acc`、`sfp_shift` |
| `reconstruct_frames()` | 从 SRAM 读 high/low 得 Y0，再读 Y0 与各 delta 结果得 Yi | `add_pmem()`、`write_sfu()` |
| `export_frame()` | 从真正的最终 SRAM 地址读数、逐项检查并导出 | `read_pmem()` |

`gather_chunk()` 不是把一个内部块独立补零。它总是使用全图坐标。例如第 64 个输出位置是 `(4,0)`，它上方需要的 `(3,0)` 数据仍来自真实上一行。只有坐标超出整张 16×16 图时才补零，因此块边界没有丢失邻居。

`xmem_word_to_l0()` 保留两个时钟：第一拍 SRAM 读地址，第二拍 L0 捕获读到的数据。OFIFO 读也分两拍：先把当前队头写入 SRAM，再等内部读指针前进。这里先采用明确、保守的时序；不会用同一拍中还没更新的数据。

## 6. RTL 为什么要改这几处

`Part2/rtl/mac.v`：同样的 `1110`，在 unsigned 模式是 14，在 signed 模式是 −2。X0 的 high/low 都用 unsigned，delta 用 signed。不能只把输入线声明为 signed 后期待两种情况都对。

`mac_tile.v`、`mac_row.v`、`mac_array.v`：`act_signed` 随 activation 一起传递。新增加的 `compute_clear` 清 activation、psum 和控制流水，但保留 `b_q` 与已经加载的状态；全局 `reset` 才清权重。

`sfp.v`：acc 时执行 `psum + (signed_input << sfp_shift)`，不 acc 时保持。移位前输入已经是 32-bit signed。把 4-bit 数直接左移四位再扩宽，会先丢掉有效位，结果错误。

`corelet.v`：连接 L0、阵列、OFIFO、SFU，并选择 SRAM 写入来自 OFIFO 还是 SFU。它不包含 SRAM。

`core.v`：增加输入/权重 SRAM 和 psum SRAM，把 corelet 接成完整仿真 DUT。TB 通过 core 端口操作；没有直接写内部答案。

所有 psum lane 用 signed 32-bit。普通 unsigned8×signed4、72 项相加的最负保守界为 `72×255×(−8)=−146880`，16-bit 不够。32-bit 留出余量；这不意味着 FPGA 上最省资源，后期可以在给出严格边界后收窄位宽。

## 7. psum SRAM 的地址怎样分

真实配置每个 bank 有 256 个 word，每个 word 有八个 signed32 lane：

| 地址 | 差分方法内容 |
|---|---|
| 0…255 | 所有 kernel 累加后的 `Conv(X0_high,W)` |
| 256…511 | `Conv(X0_low,W)` |
| 512…767 | `Conv(D1,W)` |
| 768…1023 | `Conv(D2,W)` |
| 1024…1279 | `Conv(D3,W)` |
| 1280…1535 | 当前 kernel/pass 的临时结果 |
| 1536…1791 | 重建好的 Y0，供三帧复用 |
| 1792…2047 | 当前 Yi，导出后复用于下一帧 |

普通方法把前四个 bank 用作 Y0…Y3，并在每个 high pass 累加时左移四位。两种方法有各自的完整会话，不能混用 bank 内容。

输入 SRAM 是另一块存储：当前 pass 的 256 个 word 在 0…255，零 word 在 256，九组权重从 1024 开始。完整 X0 和 deltas 保存在模拟 host/DRAM 的 TB 数组中，按当前 pass 搬进 L1。

## 8. 这次结果能说明什么

真实四帧、相同 W、相同 chunk 和输入搬运方式下：

| 指标 | 普通方法 | 差分方法 |
|---|---:|---:|
| 每个 kernel 位置的 pass | 8 | 5 |
| 整层 kernel-pass | 72 | 45 |
| chunk 执行次数 | 288 | 180 |
| 工作周期 | 225736 | 146129 |
| 四帧原始输出检查数 | 8192 | 8192 |
| 每帧 max error / MSE | 0 / 0 | 0 / 0 |
| 每个 kernel 的权重加载会话 | 1 | 1 |

工作周期减少约 35.3%，普通/差分周期比约 1.545。8/5=1.6 只是 pass 工作量比；SFU 重建、加载、排空也要花周期，所以不能直接宣称整体加速 1.6 倍。

计数包括 reset、L1 写入、权重加载、L0 gather、计算、排空、OFIFO 写回、SFU 累加/重建。额外逐项检查为 stage 90，单列排除。Python 差分生成、文件处理和 golden 生成在计数外。尚未测 FPGA Fmax、功耗或主机端到端耗时。

真实输出含负数，例如 Y0 的最小值 −11199 被完整保留。三个 delta 的饱和比例都是 `0/2048=0%`。回归另测 0/255 activation、−8/+7 delta、零差分，以及 `CHUNK=7` 时 7/7/2 的尾块；这些合成测试用于硬件边界，不是额外模型准确率。

## 9. 后期怎么改，哪些地方先别动

1. 改输入尺寸、文件、chunk：先改 `Part2/cases/teacher.json`。当前明确限制 NOUT≤256、chunk≤64，超出应先改地址图和调度，不要绕过检查。
2. 优化 SRAM→L0：`xmem_word_to_l0()` 目前每字两拍，可以预取后做流水；要同时验证首字、尾字和块边界。
3. 优化 OFIFO 写回：目前每字两拍，可以围绕真实 rd 延迟设计流水，保持 valid/读请求配对。
4. 减少重复 L1 搬运：当前两方法都在每个 kernel/pass 搬入当前输入，便于公平比较。可以扩充/重分配输入 scratchpad 后跨 kernel 保留更多数据，再用同一口径重测。
5. SFU 现在逐位置读、累加、写，周期占比最大。双 bank、更多读端口或融合移位累加都值得研究；改动后仍须检查真实 SRAM 输出与驻留权重。

视频差分本身是必做 Part2。不要直接把它重复算作 Part1 的自选 alpha。新的优化先保留本版作对照，再报告增加的硬件、工作周期和验证结果。
