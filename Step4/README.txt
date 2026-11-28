Part 1 Step 4 增量上传说明

本包只包含中文报告、验证结果、修改后的 testbench 及本说明。
基准：handoff/step2-step3，下载时 commit 1ae89946cc8f7de32e5a6278d6b9c6469980905d。

一、如何放入仓库
把压缩包解压到 ECE284_what_can_ic 仓库根目录。
仓库根目录指能同时看到 Step2、Step3、artifacts、README.md 的那一层。
解压合并后的相对路径应为：
  Step3/core_tb.v              更新已有 testbench
  Step4/report_zh.txt          新增中文验证报告
  Step4/README.txt             新增本说明
  Step4/results/              新增本次验证结果
不要把本包整体放进 Step3 或 Step4，也不要额外套一层同名仓库目录。
如你的分支已修改 core_tb.v，应先比较并合并，保留双方需要的改动。

二、包含哪些内容
Step3/core_tb.v：基于 GitHub 已有 testbench 的修改版。只修改文件名生成兼容方式（$sformatf 改为 $swrite），新增 X/Z 检查和完整比较次数检查。未修改 Step2 RTL 或七阶段控制流程。
Step4/report_zh.txt：中文报告，包含验证目标、真实输入来源、七阶段流程、实测结果、范围及复现方法。
Step4/results/team_handoff/：Step1 真实向量的编译日志、仿真日志、两次 SRAM 输出、结果摘要和阶段周期记录。
Step4/results/signed_limits/、rectangle/、kernel1/、fifo_capacity/：四组合成回归的对应证据。
Step4/results/sfp_unit/：SFU 单元测试编译与仿真日志。
Step4/results/comparison.csv：全部五组共1472个最终输出标量的逐元素比较，错误均为0。
Step4/results/verification.json：仿真器版本、基准commit、测试范围、源码SHA256及修改说明。
Step4/results/testbench_changes.patch：相对基准版本的改动记录。本包已提供修改后的core_tb.v，无需再重复应用patch。

三、哪些依赖不包含
Step2 RTL、Step3/run.py、Step3/import_handoff.py、Step3/sfp_tb.v 和 Step1 artifacts/vectors/ 均沿用原仓库，未重复打包。
未包含仿真器、本机启动包装器、训练checkpoint、编译产物或较大波形文件。
本包是叠加到现有仓库的增量包，需要原仓库依赖才能重新运行。
报告提到的 Step4/inputs/team_handoff/ 是完整复现包/本地目录中的规范化输入副本，本增量包未携带；重新运行会在 Step3/build/inputs/team_handoff/ 自动生成相同规范化数据。

四、复现
正常安装 Icarus Verilog 并保证iverilog/vvp在PATH后，在仓库根目录运行：
  python3 Step3/run.py --sim iverilog --regress --waves
生成数据和运行结果位于Step3/build/，不会自动覆盖本包保存在Step4/results/中的证据。
本次真实样本重复两次，共256个最终输出标量，错误0；加上四组合成回归共1472个最终标量，错误0。SFU单元测试通过。未执行negative-control。
报告与日志记录的是本次实测结果；没有重新训练或测量FPGA频率/功耗。

五、上传范围
向现有团队GitHub仓库提交：更新Step3/core_tb.v，新增Step4/report_zh.txt、Step4/README.txt和Step4/results/。
本次仅生成本地上传包，尚未向GitHub推送。
