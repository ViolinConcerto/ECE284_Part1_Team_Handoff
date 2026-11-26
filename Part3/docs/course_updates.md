# 2026-11-24 核对的课程更新

以下内容来自当前 Canvas 正文与 TA 回复，用来说明本工程采用哪一版要求。

| 事项 | 已核实的结论 | 来源 |
|---|---|---|
| 四帧数据 | 11 月 23 日已更换，全部 unsigned 8-bit；使用新版下载。README 明确 CHW `[8,16,16]`、channel-first、row-major | [Project Description 2026](https://canvas.ucsd.edu/courses/79115/assignments/1200550) |
| 视频输出误差 | 新四帧差分均在 signed 4-bit 范围，`Yi` 应逐项完全一致；TBD 已删除 | [Part 2 Yi tolerance](https://canvas.ucsd.edu/courses/79115/discussion_topics/1142490) |
| SFU、corelet、core | 老师只提供已有 testbench、MAC/array、SRAM、FIFO；上述剩余模块由学生完成 | [SFU/core skeleton](https://canvas.ucsd.edu/courses/79115/discussion_topics/1141875) |
| 内存导出示例 | `[HW_Code6]_Memory_Write_Read.ipynb` 已上传课程仓库；没有 HW8 | [Memory Write/Read](https://canvas.ucsd.edu/courses/79115/discussion_topics/1141824) |
| Quartus 版本 | 可使用支持指定器件的新版本；学校 Linux Cloud 提供 19.1 | [Quartus Prime version](https://canvas.ucsd.edu/courses/79115/discussion_topics/1141659) |
| Alpha 数量 | 没有固定数量，按难度与新意评估；不同方案可分别实现、评估 | [Number of alphas](https://canvas.ucsd.edu/courses/79115/discussion_topics/1141664)、[Alpha combinations](https://canvas.ucsd.edu/courses/79115/discussion_topics/1142400) |
| 展示形式 | 海报，推荐 36×27 英寸，可拼接 letter 纸；地点 Jacobs Hall 二楼 Henry Booker Room，提前 10 分钟布置。未完成 alpha 可标 In Progress | [Poster Presentation Details](https://canvas.ucsd.edu/courses/79115/discussion_topics/1142246) |

展示为 3.5 分钟讲解加 0.5 分钟问答。报告截止 12 月 12 日 23:59，最多 5 页；主要为图表时最多 6 页。展示后仍可继续完善。

正式实验读取仓库根目录的 `differential_input_batch_updated/`。旧版带负数的文件只供查阅。

老师提供的帧文件不带 W。本实验明确选择 Part1 交接的训练权重、3×3/s1/p1 卷积，配置在 `video/cases/teacher.json`；这项选择不是老师另行指定的参数。视频结果只证明所选卷积的整数运算正确，不代表四帧在完整 VGG 网络上的分类准确率。
