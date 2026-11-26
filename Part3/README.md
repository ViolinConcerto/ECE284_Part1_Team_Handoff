# Part3

七阶段 testbench，以及新版视频差分计算。Vanilla 默认接入同学交接的真实 `features.27` 数据；2024 小样例保留作回归。

## 运行

在此目录执行：

```powershell
python run_all.py --regress --waves
```

上面一条命令会导入真实模型数据，检查整数运算，再跑 Vanilla 和视频两条路径。只跑其中一条：

```powershell
python import_handoff.py
python run.py
python run.py --regress
python run.py --waves
python video/run_video.py --waves
```

工具安装和 Canvas 四帧的文件位置见 [HANDOFF.md](HANDOFF.md)。Vanilla 的 `import_handoff.py` 从仓库根目录的 `artifacts/vectors/` 生成 `cases/team_handoff/`，后续可直接重跑 `run.py`。
每个仿真默认连续运行两次任务，结果放在 `build/<case>/`。

- `simulation.log`：阶段与 PASS/FAIL。
- `cycles.csv`：各阶段周期，stage 50 是额外的原始部分和检查时间。
- `result_0.txt`、`result_1.txt`：从最终 SRAM 地址读出的结果。
- `summary.json`：参数、结果和周期记录。
- `--waves` 会生成 `core_tb.vcd`，先看顶层 `phase`、控制和 SRAM 接口。

## 先看这几处

- `tb/core_tb.v`：七阶段控制；每一阶段对应一个 task。
- `reference_dut/`：Vanilla 仿真硬件。
- `cases/team_handoff/`：从同学交接仓库验证、去除文件头后导入的真实模型样本。
- `import_handoff.py`：检查原始交接文件，重新计算整数卷积，导入测试用例。
- `video/rtl/`、`video/tb/video_tb.v`：支持两种激活符号、保留权重、SFU 移位重建的扩展硬件与控制。
- `cases/reference2024/`：固定输入、权重和原项目的 golden。
- `docs/interface.md`：Part1/2/4 的交接格式、临时控制口和后续优化位置。

`run.py` 先检查给定 golden 与独立整数卷积是否一致，再编译、仿真。TB 会检查原始部分和、累加结果以及 ReLU 结果；后两项都读 SRAM 回写后的数据检查。

代码讲解按 [vanilla_walkthrough.md](docs/vanilla_walkthrough.md) 和 [video_walkthrough.md](docs/video_walkthrough.md) 的顺序阅读。已验证结果见 [results/](results/README.md)，本机重新运行的总报告在 `build/final_report.json`。

2026-11-24 的统一回归已通过：真实层及四个 Vanilla 边界例、真实四帧及两个视频边界例、11 项 Python 检查、512 种 MAC 组合和 SFU/PE 检查。真实四帧每种方法 8192 个原始输出全部一致；工作周期为普通 225736、差分 146129，减少 35.3%。该周期口径排除主机预处理及额外检查，不是 FPGA 频率或功耗测量。

已在 Vivado 2017.4 跑过 reference2024、signed_limits、rectangle、kernel1、fifo_capacity 五例，每例连续两次任务，最终共比较 1472 个标量，均通过。合成用例用于检查硬件，不代表模型准确率。

## 参考来源与边界

Vanilla 硬件基于 2024 项目的 `Part3alpha/verilog/`。`core.v`、`corelet.v`、`sfp.v` 增加了 SFU 保持、清零与 ReLU 控制、结果回写选择和 SRAM 读出观察口。原文件 SHA-256 记录在 `docs/reference_hashes.json`。

原基线建立于 2024 参考代码。2026 年官方模板现已发布，归档与接口核对见 `docs/starter2026/`。这里把模板中的待补控制段组织成 task，并增加自动检查；参考来源见上述 hash 记录。

Vanilla 路径支持 8×8 阵列、unsigned 4-bit 激活、signed 4-bit 权重、16-bit psum。单次 padded feature map 不超过 64 个位置。

视频路径使用更新后的老师四帧（`../differential_input_batch_updated/`，CHW 8×16×16、unsigned 8-bit），权重明确选择同学训练后的 8×8×3×3 卷积权重。每次最多送 64 个输出位置，覆盖全部四帧；部分和与重建使用 signed 32-bit。旧的负数四帧保留，不能用作新版正式输入。

两个路径都只做 RTL 行为仿真。分类准确率和模型浮点误差来自 Part1 交接记录，本次不重新训练或评测全测试集；Quartus 的布局布线、频率和功耗仍需相应负责人完成。

目前先使用独立的累加和 ReLU 存储区，便于排查地址和回写。后续改调度时保留这版作对照。
