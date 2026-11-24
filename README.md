# LightMyCells v4 — Grand Challenge 容器

把一张透射光图像（BF / DIC / PC）直接预测成四个荧光通道
（Nucleus / Mitochondria / Actin / Tubulin）。

## 0. 走哪条路

平台给了 **Link GitHub Repo** 和 **Upload a Container** 两条路。

**本次走 GitHub**：本机没有 Docker（也没 WSL2），`Upload a Container` 需要先
`docker save` 出一个 ~3 GB 的 tar.gz，而服务器上的 docker daemon 我们没有权限。
GitHub 那条由平台自己 build，不依赖任何一台我们控制的机器。

```
仓库 tian101/lightmycells-gc (public)
  ├─ Dockerfile         ← 平台用这个 build；build 时从 Release 下载 4 个权重
  ├─ fetch_weights.py   REPO 已写死成 tian101/lightmycells-gc
  ├─ requirements.txt
  └─ app/               ← 推理代码
Release v1  (4 个附件，共 832 MB)
  nucleus.pth  mitochondria.pth  actin.pth  tubulin.pth
```

权重不进 git：4 × 208 MB 超过 GitHub 单文件 100 MB 上限，走 LFS 又会把
1 GB/月的带宽额度一次耗光，所以放 Release 附件，`fetch_weights.py` 在 build
时下载并 `torch.load` 校验一遍——URL 或 tag 错了会在 build 阶段就失败。

如果以后有了 Docker，`Dockerfile.local` + `weights/` 那条路仍然可用。

## 目录

```
gc_container/
├─ Dockerfile           # 平台用（从 GitHub Release 拉权重）
├─ Dockerfile.local     # 本机用（COPY weights/）
├─ fetch_weights.py     # Dockerfile 调用的下载+校验脚本
├─ requirements.txt
├─ app/                 # 训练仓库里原封不动拷过来的代码
│  ├─ gc_entrypoint.py  # ← 平台入口
│  ├─ models/           # networks.py + sfm.py
│  ├─ data/ util/ options/
│  └─ predict.py
├─ weights/             # 只有本机构建时才需要（已 gitignore）
└─ test/                # verify.py + test_inputs.py
```

## 1. 放权重

只有走 `Dockerfile.local`（本机构建）时才需要这一步：

```bash
./fetch_weights.sh                      # 默认从 SSH 别名 jupyter-gpu 拉
# 或者：CHECK_DIR=/mnt/data0/lixt/pix/check ./fetch_weights.sh   （在服务器上直接拷）
```

四个文件必须叫 `nucleus.pth` / `mitochondria.pth` / `actin.pth` / `tubulin.pth`，
共 832 MB：

| 通道 | checkpoint |
|---|---|
| nucleus | `lmc_nucleus_1111_e900_sfmfull_v4_cos1100` @1100 |
| mitochondria | `lmc_mitochondria_1111_e800_sfmfull_v2` @800 |
| actin | `lmc_actin_1118_e800_sfmfull_v4` @800 |
| tubulin | `lmc_tubulin_1118_e800_sfmfull_v4` @800 |

GitHub 路线下 4 个文件直接传成 Release `v1` 的附件，文件名保持不变。

## 1b. GitHub 上传步骤（本次采用）

顺序很重要：**先把 Release 和权重都传好，最后才在平台上 link 仓库**，否则
创建 Release 时自动生成的 `v1` tag 会触发一次注定失败的 build。

### ① 建仓库

浏览器打开 https://github.com/new

- Repository name: `lightmycells-gc`
- Public
- **不要**勾 "Add a README file" / ".gitignore" / "license"（否则本地 push 会冲突）

### ② 推代码

```powershell
cd G:\opencode\pix\gc_container
git init -b main
git add -A
git commit -m "LightMyCells v4 inference container for Grand Challenge"
git remote add origin https://github.com/tian101/lightmycells-gc.git
git push -u origin main
```

确认 `weights/` 没被提交（`.gitignore` 已排除）：

```powershell
git ls-files | Select-String weights   # 应该没有输出
```

### ③ 下载 4 个权重到本地

```powershell
$dst = "G:\opencode\pix\gc_weights"
New-Item -ItemType Directory -Force $dst | Out-Null
$ck = "/mnt/data0/lixt/pix/check"
scp jupyter-gpu:$ck/lmc_nucleus_1111_e900_sfmfull_v4_cos1100/latest_net_G.pth   $dst\nucleus.pth
scp jupyter-gpu:$ck/lmc_mitochondria_1111_e800_sfmfull_v2/latest_net_G.pth       $dst\mitochondria.pth
scp jupyter-gpu:$ck/lmc_actin_1118_e800_sfmfull_v4/latest_net_G.pth              $dst\actin.pth
scp jupyter-gpu:$ck/lmc_tubulin_1118_e800_sfmfull_v4/latest_net_G.pth            $dst\tubulin.pth
Get-ChildItem $dst | ForEach-Object { "{0}  {1:N1} MB" -f $_.Name, ($_.Length/1MB) }
```

四个文件都应该是 208.0 MB 左右。

### ④ 建 Release `v1` 并传 4 个附件

浏览器打开 `https://github.com/tian101/lightmycells-gc/releases/new`

- Tag: `v1`（新建）
- Title: `weights v1`
- 把 `nucleus.pth` / `mitochondria.pth` / `actin.pth` / `tubulin.pth`
  四个文件拖进 "Attach binaries"
- Publish release

传完后自检（必须都能返回 200，返回 404 说明文件名或 tag 不对）：

```powershell
foreach ($c in "nucleus","mitochondria","actin","tubulin") {
  $u = "https://github.com/tian101/lightmycells-gc/releases/download/v1/$c.pth"
  "{0,-14} {1}" -f $c, (Invoke-WebRequest -Method Head -Uri $u -MaximumRedirection 0 -ErrorAction SilentlyContinue).StatusCode
}
```

也可以直接在浏览器里点开其中一个 URL，确认能开始下载。

### ⑤ 在平台上 link 仓库

挑战页面 → Container Images → **Link GitHub Repo**

- 安装 Grand Challenge GitHub App，只授权 `lightmycells-gc` 这一个仓库
- 选择仓库 `tian101/lightmycells-gc`，分支/默认分支 `main`

### ⑥ push 一个 tag 触发 build

```powershell
git tag v1.0.0
git push origin v1.0.0
```

平台会从该 commit 的 `Dockerfile` 开始 build（拉 3.5 GB 基础镜像 + 832 MB
权重，通常 10–30 分钟）。构建日志在 Container Images 页面能看到。

`Dockerfile` 里的 `--tag v1` 指的是 Release，不是这个触发用的 tag，所以以后
只改代码时再 push `v1.0.1`、`v1.0.2` 即可，不必重传权重。

### 出问题怎么排

| 现象 | 原因 |
|---|---|
| build 卡在 `fetch_weights.py` | Release 没 publish，或附件名不是 `nucleus.pth` 等，或 tag 不是 `v1` |
| `is only 0.1 MB - wrong asset or an LFS pointer` | 传上去的是 git-lfs 指针文件，不是真的权重 |
| `python: can't open file '/opt/app/gc_entrypoint.py'` | `app/` 没提交进仓库（检查 `.gitignore`） |
| 运行时报 `no input images` | 平台接口的 input slug 和默认值不一致；`gc_entrypoint.py` 会从 interface.json 自动读 slug，若报这个说明挂载路径约定不同 |

## 2. 构建（在有 Docker 的机器上）

```bash
./build_and_save.sh          # 构建 + 冒烟测试 + 导出 lightmycells-v4.tar.gz
```

等价于：

```bash
docker build -f Dockerfile.local --platform linux/amd64 -t lightmycells-v4:1 .
docker save lightmycells-v4:1 | gzip -1 > lightmycells-v4.tar.gz
```

如果本机没有 Docker，跳过后面的 §2/§3/§4，直接看 §1b 走 GitHub。

镜像约 4–5 GB，压缩后约 3 GB。首次构建要拉
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`（约 3.5 GB）。

## 4. 本地跑一遍（不用平台上）

```bash
mkdir -p /tmp/gc/input/images/organelles-transmitted-light-ome-tiff
cp /path/to/some_field.tif /tmp/gc/input/images/organelles-transmitted-light-ome-tiff/abc-123.tif

docker run --gpus all --rm \
  -v /tmp/gc/input:/input:ro \
  -v /tmp/gc/output:/output \
  lightmycells-v4:1
```

期望输出：

```
/output/images/nucleus-fluorescence-ome-tiff/abc-123.tif
/output/images/mitochondria-fluorescence-ome-tiff/abc-123.tif
/output/images/actin-fluorescence-ome-tiff/abc-123.tif
/output/images/tubulin-fluorescence-ome-tiff/abc-123.tif
```

都是单通道 `uint16` OME-TIFF，尺寸与输入一致，文件名沿用输入的 uuid。

## 5. 接口与实现要点

| 项 | 值 |
|---|---|
| 输入 | `/input/images/organelles-transmitted-light-ome-tiff/<uuid>.tif` |
| 输出 | `/output/images/<channel>-fluorescence-ome-tiff/<uuid>.tif` |
| 归一化 | 整幅图 1% / 99.5% 分位拉伸到 uint8（与训练一致） |
| 推理 | 512 px 分块、步长 384、Hann 窗融合（与评测一致） |
| 输出 dtype | uint16，`0..255 → 0..65535` 线性映射 |
| 架构 | `unet_256`, `ngf 64`, `norm batch`, `sfm_levels 5 / start 2 / msau 1 / fan 1`，`input_nc=output_nc=1` |

入口会被平台以 `gc_entrypoint.py <interface.json> <output_dir>` 调用。两个参数都
可以接受也可以忽略：读写路径固定为 `/input` 和 `/output`（本地测试用 `INPUT_DIR`
/ `OUTPUT_DIR` 覆盖）。若 `interface.json` 存在，会从中读出真实的 slug，因此平台上
改 slug 不需要重打镜像。

输入兼容：

- 单张 2D 灰度
- z-stack（`Z, H, W`）——默认取 Laplacian 方差最大的那张（`--z_select sharp`）；
  `--z_select all` 则在每张上都推理、保留最清晰的结果
- 彩色图 / `(Z, H, W, C)` ——按 Rec.601 灰度化
- `.mha / .mhd` ——需要 SimpleITK。Dockerfile 里是**尽力安装**（装不上只会退化，
  不会让 build 失败）；纯 `.tif` 路径不依赖它

推理耗时（A100 80GB，2044×2048 单视野）：首次含热身约 5 s，之后每通道约 0.16 s，
一个 case 合计约 6 s。CPU 上慢 100 倍以上，平台上请申请 GPU。

## 6. 自测（在训练服务器上跑）

`test/verify.py` 会做三件事：

1. 断言 `gc_entrypoint.normalise()` 与 `data/lmc_dataset.normalise()` 对
   uint8 / uint16 / float32 输出逐位相同；
2. 断言容器重建的网络权重与 `predict.load_generator()` 完全一致，且
   `tiled_infer()` 与 `predict.infer_image(..., mode="tile")` 的预测数组
   `max|diff| == 0`；
3. 用真实测试视野伪造 `/input` 目录跑一遍入口，检查四个输出文件名、尺寸、
   dtype。

`test/test_inputs.py` 另外覆盖输入形态（z-stack / RGB / 301×407 奇数尺寸 /
一次多个 case / `.mha`）并给出与 GT 对比的指标。

### 已验证结果（2024-11-24，A100）

```
normalise[u8  ] identical: True     normalise[u16 ] identical: True
normalise[f32 ] identical: True     normalise[real] identical: True
weights: 219 tensors, same count=True, identical=True
prediction: max|diff|=0.000e+00  (shape (1200, 1200))

  plain 2D               OK        z-stack (4 planes)     OK
  colour RGB             OK        odd 301x407            OK
  two inputs             OK

quality vs ground truth (Study_18_img00274, v4 weights):
  nucleus       PSNR  20.60 dB   Pearson 0.911   SSIM 0.760
  mitochondria  PSNR  15.45 dB   Pearson 0.479   SSIM 0.445
```

平台式调用（只给两个位置参数、`/output` 不存在时退回 `$2`、2044×2048 非 256
整数倍）也跑通了，四个输出都正确落盘。

```bash
# 服务器上
mkdir -p /tmp/gc && cp -r app /tmp/gc/app && mkdir -p /tmp/gc/weights
cd /mnt/data0/lixt/pix/check
ln -sf $PWD/lmc_nucleus_1111_e900_sfmfull_v4_cos1100/latest_net_G.pth      /tmp/gc/weights/nucleus.pth
ln -sf $PWD/lmc_mitochondria_1111_e800_sfmfull_v2/latest_net_G.pth          /tmp/gc/weights/mitochondria.pth
ln -sf $PWD/lmc_actin_1118_e800_sfmfull_v4/latest_net_G.pth                 /tmp/gc/weights/actin.pth
ln -sf $PWD/lmc_tubulin_1118_e800_sfmfull_v4/latest_net_G.pth               /tmp/gc/weights/tubulin.pth

cd /mnt/data0/lixt/pix/pytorch-CycleGAN-and-pix2pix-master
/usr/local/anaconda3/envs/pytorch/bin/python /tmp/gc/test/verify.py
```

## 7. 已知情况（提交前请知悉）

- 四个权重里都带 SFM 模块（`sfm_levels=5 ...`）。我们在 800 轮的消融实验里已经
  证实 **SFM 的 ARS 采样网格始终是恒等变换**（位移均值/中位数/99 分位都是 0.00 px），
  也就是说这四个模型在数值上等价于普通 U-Net。装进容器不会有正确性风险，只是
  多了不参与计算的参数；但**不要对外宣称 v4 比 U-Net 更好**。
- 细胞核上 v4 实测 21.66 dB，纯 U-Net（`lmc_nucleus_1123_e800_abl_base`）22.62 dB，
  差异在 ±0.5 dB 的初始化噪声内。若想换成纯 U-Net，只需替换
  `weights/nucleus.pth`，并把 `ARCH` 里的 `sfm_*` 置 0。
- CPU 上推理会非常慢（单视野单通道约 30–60 s），平台上请申请 GPU。
