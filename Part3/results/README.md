# 已验证结果

| 用例 | 验证结果 |
|---|---|
| 真实 `features.27`，Vanilla | 两次任务，256 个最终标量全部一致 |
| 老师新版四帧，普通与差分 | 每种方法各 8192 个原始卷积输出，逐帧最大误差和 MSE 均为 0 |
| 两组视频边界用例 | 0/255、−8/+7、零差分与不足一块的尾部数据通过 |

仿真工作周期 225,736 → 146,129，减少 35.3%。主机预处理和 testbench 检查时间另计。[柱状图 PNG](cycle_comparison.png) · [SVG](cycle_comparison.svg)

在仓库根目录执行 `python Part3/run_all.py --regress`，本机完整报告会写入 `Part3/build/final_report.json`。视频逐帧结果、周期和权重加载记录在 `Part3/video/build/teacher_updated/`。
