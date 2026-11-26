# Video differential processing

在 `Part3` 目录运行：

```powershell
python video/run_video.py --regress --waves
```

`python video/run_video.py --prepare-only` 只准备数据。`video/cases/teacher.json` 从仓库根目录读取四帧和 Part1 权重。四帧下载位置和仿真器配置见 [小组交接](../HANDOFF.md)。

输入采用老师 11 月 23 日更新的 unsigned 四帧；旧负数数据会被拒绝。W 明确选择同学 Part1 的训练权重。所有输出都是 **ReLU 前的原始整数卷积**，不乘模型 scale。

| 文件 | 作用 |
|---|---|
| `run_video.py` | 读取 CHW 帧，准备 unsigned nibbles / signed deltas，独立计算 golden，编译仿真，验证 SRAM 导出结果 |
| `cases/teacher.json` | 尺寸、路径、W 来源与 chunk 大小 |
| `tb/video_tb.v` | 控制 SRAM、L0、PE、OFIFO、SFU；实际执行与重建在 RTL 中完成 |
| `rtl/` | Part1 数据通路的扩展版，支持符号模式、保留权重的清流水，以及 32-bit 部分和 |
| `tests/rtl_units_tb.v` | MAC、PE 权重保持、SFU 测试 |
| `tests/test_vectors.py` | 独立整数、打包和非法输入检查 |

`build/teacher_updated/summary.json` 包含逐帧误差、输入统计、周期、权重加载记录和源码哈希。`conventional_frame0..3.txt` / `differential_frame0..3.txt` 是实际 SRAM 输出。`cycles.csv` 中 stage 90 是额外检查开销，不计入工作周期。`video_tb.vcd` 可按 method、kernel_index、pass_index、phase 定位计算。

每个 kernel 位置加载一次 W，常规方法随后运行八个 pass，差分方法运行五个；同一位置的 W 跨全部 pass 和空间 chunk 保留。3×3 卷积共有九个位置，不能把整层的阵列执行次数说成五次。

完整讲解见 `../docs/video_walkthrough.md`。这里没有 Quartus 布局布线或功耗测量；仿真时钟周期和测得的工作周期不能当作 FPGA Fmax。
