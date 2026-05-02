# 处理 DREAMS 数据

这份文档记录如何把 DREAMS 的 frame-major 数据整理成 EasyVolcap / 4K4D 可以直接使用的格式，并继续生成 `vhulls` 和 `surfs`。

## 处理流程

1. 重排图像和 mask。

```shell
python scripts/dreams/rearrange_frameset_seq.py \
  --src data/dreams/take2/frameset/color_frames \
  --dst data/dreams/take2_rearranged \
  --rgba-foreground-only
```

这一步会把 DREAMS 原始的 frame-major 结构：

```text
color_frames/
  000000/
    00.png
    01.png
    ...
  000001/
    00.png
    01.png
```

整理成 EasyVolcap 需要的 camera-major 结构：

```text
data/dreams/take2_rearranged/
  images/
    00/
      000000.png
      000001.png
    01/
      000000.png
      000001.png
  masks/
    00/
      000000.png
      000001.png
```

在 `--rgba-foreground-only` 模式下，`images/` 保存合成到背景色上的 RGB 图像，`masks/` 保存从 RGBA alpha 通道提取出来的单通道 PNG mask。

2. 构建 `intri.yml` 和 `extri.yml`。

```shell
python scripts/dreams/cameras_json_to_easymocap_intri_extri.py \
  --src data/dreams/take2/frameset/color_frames \
  --dst data/dreams/take2_rearranged \
  --overwrite
```

这一步会读取 DREAMS 的 `cameras.json`，并在目标目录写出：

```text
data/dreams/take2_rearranged/
  intri.yml
  extri.yml
```

3. 写 `configs/datasets/dreams` 下对应 sequence 的配置参数。

例如 `take2_rearranged` 对应：

```text
configs/datasets/dreams/dreams.yaml
configs/datasets/dreams/take2_rearranged.yaml
```

其中 `dreams.yaml` 放 DREAMS 数据集通用参数，`take2_rearranged.yaml` 指定当前 sequence 的 `data_root` 和 `images_dir`。

4. 生成 `vhulls` 和 `surfs`。

```shell
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/dreams/take2_rearranged.yaml,configs/specs/vhulls.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/dreams/take2_rearranged.yaml,configs/specs/surfs.yaml
```

第一条命令会根据 `images/`、`masks/`、`intri.yml`、`extri.yml` 做 visual hull space carving，并输出 `vhulls/`。日志里会打印 accumulated bounding box，可以把它复制到对应的 `_obj.yaml` 里作为更紧的训练范围。

第二条命令会把 `vhulls/` 进一步处理成 `surfs/`，用于后续 4K4D 初始化或训练。

生成完成后需要可视化检查 `vhulls` 和 `surfs` 是否合理。重点看每一帧的人体/物体轮廓是否完整、是否有明显漂浮噪点、是否被过度裁切，以及 `surfs` 相比 `vhulls` 是否保留了主体结构。

```shell
python scripts/tools/view_ply.py data/dreams/take2_rearranged/vhulls

python scripts/tools/view_ply.py data/dreams/take2_rearranged/surfs
```

可视化窗口中使用左右方向键切换同目录下的 `.ply` 文件。

5. 训练 4K4D 模型。

```shell
evc-train -c configs/exps/4k4d/4k4d_take2_rearranged_r4.yaml \
  exp_name=4k4d_take2_rearranged_r4
```

训练记录会写到 `data/record/4k4d_take2_rearranged_r4/`，验证和渲染指标会写到 `data/result/4k4d_take2_rearranged_r4/`。

6. 生成支持实时渲染的 super charged 版本。

```shell
python scripts/realtime4dv/charger.py --sampler SuperChargedR4DV --exp_name 4k4d_take2_rearranged_r4 -- -c data/record/4k4d_take2_rearranged_r4/4k4d_take2_rearranged_r4_1777440586.yaml,configs/specs/super.yaml
```

这一步会把训练好的模型转换成 `SuperChargedR4DV` 推理版本，提前缓存实时渲染需要的点特征、颜色混合和几何参数，方便后续用 viewer 交互渲染。

转换完成后可以启动 GUI 查看：

```shell
evc-gui -c data/record/4k4d_take2_rearranged_r4/4k4d_take2_rearranged_r4_1777440586.yaml,configs/specs/superf.yaml,configs/specs/vf0.yaml exp_name=4k4d_take2_rearranged_r4
```

如果需要直接测试渲染单个原始相机视角，并保存 22 fps 视频，可以用：

```shell
evc-test -c data/record/4k4d_take2_rearranged_r4/4k4d_take2_rearranged_r4_1777440586.yaml,configs/specs/superf.yaml,configs/specs/eval.yaml \
  exp_name=4k4d_take2_rearranged_r4 \
  val_dataloader_cfg.sampler_cfg.view_sample=1,2,1 \
  runner_cfg.visualizer_cfg.store_video_output=True \
  runner_cfg.visualizer_cfg.video_fps=22 \
  runner_cfg.visualizer_cfg.save_tag=cam01_22fps
```

这里 `val_dataloader_cfg.sampler_cfg.view_sample=0,1,1` 表示只渲染原始相机 `cam00`。`configs/specs/superf.yaml` 使用 super charged 实时推理版本，`configs/specs/eval.yaml` 只输出 `RENDER` 结果。README 中的 `configs/specs/spiral.yaml` 和 `configs/specs/ibr.yaml` 用于 novel-view spiral path 渲染，不是单个原始相机视角必须的配置。

## 重排数据 `scripts/dreams/rearrange_frameset_seq.py`

这个脚本负责把 DREAMS 的逐帧目录转换成 EasyVolcap 使用的逐相机目录。

默认输入结构：

```text
color_frames/
  000000/
    00.png
    01.png
  000001/
    00.png
    01.png
```

默认输出结构：

```text
data/dreams/seq0/
  images/
    00/
      000000.png
      000001.png
  masks/
    00/
      000000.png
      000001.png
```

### RGBA 前景数据

如果源数据是 RGBA 前景图，使用：

```shell
python scripts/dreams/rearrange_frameset_seq.py \
  --src data/dreams/take2/frameset/color_frames \
  --dst data/dreams/take2_rearranged \
  --rgba-foreground-only
```

这个模式下：

- `images/` 保存 RGB 图像；
- `masks/` 保存 alpha-only PNG mask；
- 默认背景色是黑色。

修改合成背景色：

```shell
python scripts/dreams/rearrange_frameset_seq.py \
  --rgba-foreground-only \
  --background 255,255,255
```

### 原始图像和 mask 分开存放

如果原始图像和 mask 已经分别在两个 frame-major 目录下，使用：

```shell
python scripts/dreams/rearrange_frameset_seq.py \
  --src path/to/raw_color_frames \
  --mask-src path/to/mask_frames \
  --dst data/dreams/seq0
```

这会把图像写到 `images/`，把 mask 写到 `masks/`。

### 常用选项

只预览，不写入：

```shell
python scripts/dreams/rearrange_frameset_seq.py --dry-run
```

覆盖已有文件：

```shell
python scripts/dreams/rearrange_frameset_seq.py --overwrite
```

把输出帧号重新编号成连续编号：

```shell
python scripts/dreams/rearrange_frameset_seq.py --renumber
```

关闭进度条：

```shell
python scripts/dreams/rearrange_frameset_seq.py --no-progress
```

把相机目录直接放在输出根目录，而不是 `images/` 下：

```shell
python scripts/dreams/rearrange_frameset_seq.py --images-dir .
```

## 提取相机内外参 `scripts/dreams/cameras_json_to_easymocap_intri_extri.py`

这个脚本负责把 DREAMS 的 `cameras.json` 转成 EasyMocap / EasyVolcap 使用的相机文件：

```text
intri.yml
extri.yml
```

`take2` 的默认用法：

```shell
python scripts/dreams/cameras_json_to_easymocap_intri_extri.py \
  --src data/dreams/take2/frameset/color_frames \
  --dst data/dreams/take2_rearranged \
  --overwrite
```

输出文件会写到：

```text
data/dreams/take2_rearranged/
  intri.yml
  extri.yml
```

`--src` 可以传入某一帧的 `cameras.json`，也可以传入某个 frame 目录，或者直接传入 `color_frames` 根目录。传入 `color_frames` 根目录时，脚本会使用排序后的第一个 `*/cameras.json`。

脚本默认会从 `data/dreams/take2_rearranged/images` 推断相机名字，使相机名和 `images/00`、`images/01` 这样的目录一致。如果目标目录还没有 `images/`，则使用从 `00` 开始的补零编号。

常用选项：

```shell
# 使用 K_00 / D_00，而不是 K_00 / dist_00
python scripts/dreams/cameras_json_to_easymocap_intri_extri.py --dist-key D

# 目标目录还没有 images/ 时，使用四位相机编号
python scripts/dreams/cameras_json_to_easymocap_intri_extri.py --name-digits 4
```
