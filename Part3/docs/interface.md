# 接口与读代码顺序

## 数据交接

每个 case 放四个文件，使用 `python run.py --case <目录>`：

| 文件 | 内容 |
|---|---|
| `case.json` | H/W、kernel、padding、stride、位宽和通道数，参考现有样例 |
| `input.txt` | 补零后的输入；空间按行排列，每行 32 个二进制位，channel 0 在低 4 位 |
| `weight.txt` | 顺序为 kernel 行→kernel 列→output channel；每行 32 bit，input channel 0 在低 4 位 |
| `psum.txt` | 完整卷积加 ReLU 后的整数 golden；输出空间按行排列，每行 128 bit，output channel 0 在低 16 位 |

允许空行以及以 `#` 或 `//` 开头的说明行。权重、psum 使用二补码。输入 padding 必须是零。

当前是 groups=1、dilation=1、无 bias。尺度恢复由 Part1/4 在软件侧处理，不能把浮点结果直接放进整数 golden。
正式数据还需要提供 checkpoint、样本编号、目标模块名和 scale，便于把模型与文件对应起来；2024 样例缺少这些信息，不能据此声称完成了本组 Part4。

## Part2 临时适配口

`inst[33:0]` 保留旧格式：

| 位 | 控制 |
|---|---|
| 33 | SFU accumulate |
| 32 / 31 / 30:20 | psum SRAM CEN / WEN / address |
| 19 / 18 / 17:7 | input SRAM CEN / WEN / address |
| 6 | OFIFO read |
| 5:4 | 保留为零，不使用 IFIFO |
| 3 / 2 | L0 read / write |
| 1 / 0 | execute / weight load |

为满足今年回写要求，参考 DUT 另加三个控制：`sfp_clear`、`sfp_relu`、`pmem_from_sfp`。
`acc=0` 且 clear/relu 都为零时保持原值。clear、acc、relu 不同时使用。
`pmem_from_sfp=0` 写 OFIFO 数据，=1 写 SFU 数据；写使能和地址仍由 SRAM 控制字段给出。

`pmem_read_data` 用于从 core 端口检查真实 SRAM 内容，`l0_ready`、`ofifo_full` 用于基本流控检查。接小组 Part2 后，先对齐这些功能，不要求对方照搬端口命名。

## 跟一遍 core_tb

1. `load_scratchpad`：写入输入和全部 kernel 权重，TB 模拟外部 DRAM。
2. `load_pe_kernel`：读 SRAM 到 L0，再同时启动 L0 读和 PE load。
3. `load_l0_activations`：读输入到 L0。
4. `execute_kernel`：发 NIJ 拍 execute，随后停止新任务并排空流水。
5. `store_partials`：先把 OFIFO 当前头写入 SRAM，再让 FIFO 弹出。内部 read enable 有一拍寄存，不能同拍随意连续读。
6. `accumulate_outputs`：按卷积空间位置读取各 kernel 部分和，SFU 累加，写回带符号结果。
7. `relu_outputs`：读回累加结果，送入已清零的 SFU，再 ReLU 并写回。

`tick()` 等待上升沿采样，再返回下一个下降沿。控制都在下降沿改变，避免 TB 与 RTL 在上升沿竞争。
同步 SRAM 第一个时钟锁存读地址，第二个时钟才让 L0/SFU 使用这份数据。读和消费之间 SFU 保持，不会提前做 ReLU。

## 地址布局

输入 SRAM：输入从 0 开始，权重从 1024 开始，每个 kernel 占 8 word。

psum SRAM：

```text
raw  : [0, NK*NIJ)
sum  : [NK*NIJ, NK*NIJ + NOUT)
relu : [NK*NIJ + NOUT, NK*NIJ + 2*NOUT)
```

2024 样例是 raw 0..323、sum 324..339、relu 340..355。
输出 `(oy, ox)` 的 kernel `(ky, kx)` 部分和地址为：

```text
(ky*KERNEL+kx)*NIJ + (oy*STRIDE+ky)*PAD_W + ox*STRIDE+kx
```

## 后续可改的地方

先保留当前结果检查，再单独改变一个调度点：

- SRAM→L0 读写流水化：当前每字分两拍，便于确认读延迟。
- 执行与 OFIFO 写回重叠：当前先算后存，方便验证；改后比较队列峰值和总周期。
- SFU 连续读取部分和：当前每次读完再累加，后续可减少空拍。
- 复用 sum/relu 地址：需要先证明没有覆盖仍在使用的数据。

`cycles.csv` 分开记录各阶段及额外检查时间。它是仿真控制周期记录，尚不能当作 FPGA 频率、功耗或吞吐结果。
