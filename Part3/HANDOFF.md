# 给组员的交接

小组工程把 Part1 训练、七阶段 testbench、视频差分 RTL 和 Quartus 工程放在一个仓库。两条仿真路径都读取 [`artifacts/vectors/`](../artifacts/vectors/) 里同一份训练权重。

## 怎么跑

从[课程 Canvas](https://canvas.ucsd.edu/courses/79115/assignments/1200550)下载更新后的 unsigned 四帧，把 `X0.txt` 到 `X3.txt` 解压到仓库根目录的 `differential_input_batch_updated/`。这四帧放在各自电脑的本地数据目录。

在仓库根目录执行：

```powershell
python Part3/run_all.py --regress
```

需要 Python 3.10+ 与 Vivado 或 Icarus Verilog。Vivado 安装目录不在 PATH 时，加 `--sim vivado --vivado-bin "实际路径"`。要保存波形时加 `--waves`。新结果写入 `Part3/build/` 和 `Part3/video/build/`。

## 接了哪些数据，结果是什么

- Part1：同学交接的 `features.27`，8→8、3×3；真实样本两次运行，256 个最终输出全部一致。
- 视频：老师 11 月 23 日更新的 unsigned 四帧，CHW 8×16×16；沿用上述训练权重，计算 ReLU 前的卷积结果。
- 普通、差分两种方法各 8192 个输出，全部零误差；三份差分无饱和。
- 相同搬运方式下，工作周期 225736→146129，减少 35.3%。主机生成差分和额外检查不计入，尚未测 FPGA 频率／功耗。

结果和对比图在 [results/](results/README.md)。

## 代码找哪里

| 要接的东西 | 文件 |
|---|---|
| 七阶段 testbench | `tb/core_tb.v` |
| Vanilla 的硬件 | `reference_dut/` |
| 视频控制、硬件 | `video/tb/video_tb.v`、`video/rtl/` |
| 视频输入、权重与分块配置 | `video/cases/teacher.json` |
| Quartus 工程 | [`fpga/vanilla_corelet.qpf`](fpga/vanilla_corelet.qpf)，顶层为 `corelet` |

## 现在需要分工

1. **Quartus**：完成布局布线，拿到 slow-corner 频率和 20% 输入活动率下的功耗。工程已准备，尚未实跑。
2. **自选 alpha**：确定方案与负责人。视频差分是必做项，不能重复算作 alpha；选定方案后与本版对照验证。
3. **海报、报告**：12 月 1 日海报展示，3.5 分钟讲解＋0.5 分钟问答；海报推荐 36×27 英寸，提前 10 分钟布置。报告 12 月 12 日截止。

代码讲解在 `docs/`，课程更新在 `docs/course_updates.md`。Part1 的 90.94% 准确率来自交接的训练记录；这次新增验证证明 RTL 运算正确，没有重新评测完整网络。

## 没有共享仓库时

已有 Part1 仓库的组员可以用离线文件 `ECE284_RTL_sync.bundle` 导入合并提交。下面假设文件放在仓库目录里：

```text
git fetch ./ECE284_RTL_sync.bundle refs/heads/integration/rtl-video
git switch -c integration/rtl-video FETCH_HEAD
```

离线文件依赖原 Part1 提交 `914e159f`。以后如果建了共享仓库，把 `integration/rtl-video` 推到共享仓库并开 PR 即可。
