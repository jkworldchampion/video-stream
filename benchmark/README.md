# Benchmark

## Prepare Datasets
Download datasets from the following links:
[Sintel](http://sintel.is.tue.mpg.de/), [KITTI](https://www.cvlibs.net/datasets/kitti/), [Bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [ScanNet](http://www.scan-net.org/), [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

```bash
pip3 install natsort
cd benchmark/dataset_extract
python3 dataset_extrtact${dataset}.py
```
This script will extract the dataset to the `benchmark/dataset_extract/dataset` folder. It will also generate the json file for the dataset.

## Run inference
```bash
python3 benchmark/infer/infer.py \
    --infer_path ${out_path} \
    --json_file ${json_path} \
    --datasets ${dataset}
```
Options:
- `--infer_path`: path to save the output results
- `--json_file`: path to the json file for the dataset, like `sintel_video.json`, `scannet_video_500.json`, `scannet_video_tae.json`
- `--datasets`: dataset name, choose from `sintel`, `kitti`, `bonn`, `scannet`, `nyuv2`

## Run evaluation
```bash
## tae
bash benchmark/eval/eval_tae.sh ${out_path} benchmark/dataset_extract/dataset
## ~110frame like DepthCrafter
bash benchmark/eval/eval.sh ${out_path} benchmark/dataset_extract/dataset
## ~500frame 
bash benchmark/eval/eval_500.sh ${out_path} benchmark/dataset_extract/dataset
```

## Scale Drift Profiling
Use `benchmark/eval/scale_drift_profile.py` to quantify per-frame scale/shift drift against ground truth.

Example (streamed predictions):
```bash
python benchmark/eval/scale_drift_profile.py \
  --pred-root outputs_streaming/scannet_stream_valmini \
  --json benchmark/datasets/scannet/scannet_video_500.json \
  --dataset-key scannet \
  --dataset-eval-tag scannet_500 \
  --output logs/scale_drift_stream.json \
  --prefix stream
```

Run the same command with the clip predictions (e.g. `outputs/clip_scannet_valmini`) and a different `--prefix` to compare summaries. The JSON report includes per-frame optimal scale/shift, cumulative standard deviation curves, and optional segment-wise stats when `--segments-json` is provided (format: `{ "scene_id": [{"name": "...","start":0,"end":47}, ...] }`).
