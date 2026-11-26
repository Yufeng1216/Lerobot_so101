# 远程训练

[English](README.md) · **简体中文**

`scripts/` 里的东西都在 Windows 笔记本上跑（机械臂、录数据、本地训练、部署）。
这个文件夹放的是在**别的机器上**跑的东西，给笔记本 8 GB 显存放不下的策略用。

| 文件夹 | 在哪跑 | 用来训 |
|---|---|---|
| `slurm/` | 用 Slurm 排队系统的 Linux GPU 集群 | 默认大小的 Diffusion Policy（2.78 亿参数）；batch 32 的 SmolVLA |
| `hf_jobs/` | 租用的 Hugging Face GPU（按秒计费） | π₀.₅（41 亿可训练参数，float32 要 80 GB 的卡） |

## `slurm/`

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 在登录节点上一次性装环境（版本和笔记本一致） |
| `site.env.example` | 账号 / 分区 / 时长上限的模板；复制成 `site.env`（不进 git） |
| `submit.sh` | 检查账号是否在允许列表里，然后提交 `train.sbatch` |
| `train.sbatch` | 任务本身：激活环境、下载数据集、按策略预设运行 `lerobot-train`；再次提交时自动续训 |

```bash
# 在笔记本上，仓库根目录
scp -r cloud <用户名>@<集群登录节点>:~/Lerobot/

# 在集群上
cd ~/Lerobot
bash cloud/slurm/setup_env.sh                      # 只需一次
cp cloud/slurm/site.env.example cloud/slurm/site.env && nano cloud/slurm/site.env
bash cloud/slurm/submit.sh diffusion             # 或者 smolvla
squeue -u $USER
tail -f logs/diffusion_<任务编号>.log
```

任务到了时长上限，再运行一次同样的 `submit.sh` 命令，就会从上一个 checkpoint 接着训。训完把部署要用的模型拷回笔记本：

| 策略 | 要拷的文件夹 |
|---|---|
| diffusion | `outputs/train/diffusion_so101_pick_place/checkpoints/last/pretrained_model_ema/` |
| smolvla | `outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model/` |

PS：SmolVLA 第一次用时要从 Hub 下载 `lerobot/smolvla_base`（约 2 GB），所以计算节点要能上网。

Linux 环境的几处处理（脚本里都做好了）：

- PyTorch 用 CUDA 12.6 版：CUDA 12.8 版已经不带 V100 的 GPU 程序了。
- 视频用 PyAV 解码（`--dataset.video_backend=pyav`）；torchcodec 的 Linux 安装包是给 CUDA 13 编译的。
- `LD_LIBRARY_PATH` 让 conda 的 C++ 运行库排在系统的前面。

## `hf_jobs/`

`train_pi05.py` 从 Windows 笔记本把 π₀.₅ 的微调任务提交到 [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)，
用的是 LeRobot 自带的 `--job.target=<硬件>`。

```bash
python cloud/hf_jobs/train_pi05.py --dry-run                     # 只打印任务配置，不提交
python cloud/hf_jobs/train_pi05.py --steps 200 --timeout 40m     # 测速度和显存
python cloud/hf_jobs/train_pi05.py --steps 3000 --timeout 9h     # 正式训练
```

第一次之前要准备：

- 一个有 **write** 权限的 Hugging Face token（细粒度 token 还要勾上 Jobs 权限），账号里要有余额；
- 同意 [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) 的许可：π₀.₅ 用它的分词器。

在 `a100-large`（$2.50/小时）上：batch 32 每步约 8.3 秒（float32 全量微调，开梯度检查点），显存 66.9 GB。
默认的 3 万步要约 70 小时，所以只训 3000 步，学习率曲线按新步数缩短，其余全用 LeRobot 默认。`--timeout` 不算排队等 GPU 的时间。

PS：在 Windows 上，LeRobot 会把模型 id 写成 `lerobot\pi05_base` 放进任务配置；脚本在上传配置前把它改回斜杠，不然 Linux 上的任务找不到模型。

3000 步用了 7 小时 6 分（约 $18），最终 loss 0.008，模型推到了任务创建的私有仓库。训练曲线、下载命令和部署方案
（笔记本显存放不下，推理要通过 `lerobot-policy-server` 放到租的 GPU 上）见 [docs/07_pi05.zh-CN.md](../docs/07_pi05.zh-CN.md)。

PS：`hf jobs ps` / `hf jobs logs` 会输出 Windows GBK 终端编码不了的字符，先设 `PYTHONUTF8=1`（`scripts/env.ps1` 已经设了）。
