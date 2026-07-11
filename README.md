<p align="center">
  <h1 align="center"><ins>OpenFrontier:</ins><br>General Navigation with Visual-Language Grounded Frontiers</h1>
  <p align="center">
    <a href="https://esteb37.github.io">Esteban&nbsp;Padilla-Cerdio</a><sup>*</sup>
    ·
    <a href="https://boysun045.github.io/boysun-website/">Boyang&nbsp;Sun</a><sup>*</sup>
    ·
    <a href="https://cvg.ethz.ch/team/Prof-Dr-Marc-Pollefeys">Marc&nbsp;Pollefeys</a>
    ·
    <a href="https://hermannblum.net/">Hermann&nbsp;Blum</a>
  </p>
  <p align="center"><sup>*</sup> equal contribution</p>
  <h2 align="center">
    <p>RSS 2026</p>
    <a href="https://arxiv.org/abs/2603.05377" align="center">ArXiv</a> |
    <a href="https://boysun045.github.io/OpenFrontier-Project/" align="center">Webpage</a>
  </h2>
</p>

OpenFrontier is a zero-shot navigation framework that treats navigation as sparse
subgoal identification and reaching. It uses visual frontiers as semantic
anchors for vision-language priors, enabling goal-conditioned navigation without
task-specific policy training or fine-tuning.

## Quick Start

- [Setup](#setup) - Choose the Open3D demo or Habitat benchmarking pipeline.
- [Navigation Demos](#navigation-demos) - Run Open3D navigation demos.
- [Habitat Benchmarking](#habitat-benchmarking) - Evaluate HM3D, MP3D, and OVON.
- [Pipeline Configuration](#pipeline-configuration) - Tune mapping, planning, and VLM settings.

## Setup


Clone the repository recursively so third-party modules are available:

```bash
git clone --recursive git@github.com:cvg/OpenFrontier.git openfrontier
cd openfrontier
```

### SAM3

If you did not clone recursively, initialize SAM3:

```bash
git submodule update --init --recursive third_party/sam3
```

SAM3 is included under `third_party/sam3`. Follow the upstream
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) installation
instructions for SAM3-specific environment and checkpoint requirements.

#### Installation
```bash
cd third_party/sam3
conda create -n sam3 python=3.12 -y
conda activate sam3
python -m pip install --force-reinstall "setuptools<82"
pip install -e .
pip install -e ".[notebooks]"
pip install flask
cd ../..
```

If starting the server fails with `RuntimeError: The NVIDIA driver on your
system is too old`, the default PyTorch wheel was built for a newer CUDA than
your driver supports. Install a build matching your driver's CUDA version
(shown by `nvidia-smi`), e.g. for CUDA 12.8:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### Run server in second window
From the repository root:

```bash
conda activate sam3
python sam3_server.py
```

SAM3 checkpoints are downloaded from Hugging Face on first use; request access to the
`facebook/sam3` checkpoint repository and authenticate before starting
`sam3_server.py`:

```bash
hf auth login
```

### OpenFrontier Environments

OpenFrontier has two installation paths:

- [Open3D demo pipeline](#open3d-demo-pipeline): lightweight local demos with
  Python 3.11 and no Habitat dependency.
- [Habitat benchmarking pipeline](#habitat-benchmarking-pipeline): HM3D, MP3D,
  and OVON evaluation using the Habitat-Lab/Habitat-Sim conda environment.

## Open3D Demo Pipeline

Use this pipeline for `demo_navigation.py`, `demo_exploration.py`, and
single-image debugging. It does not install Habitat.

Create and activate the Python 3.11 environment:

```bash
conda create -n openfrontier python=3.11 -y
conda activate openfrontier
```

Install the Open3D demo dependencies and download model weights:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -r requirements.txt
bash scripts/download_weights.sh
```


### Optional editable installs:

```bash
pip install -e third_party/UniK3D
pip install -e third_party/Metric3D
```

If using Metric3D as a depth source, install mmcv with setuptools resolution

```bash
echo "setuptools<82" > /tmp/pip-constraints.txt
PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir mmcv
```

### VLM Setup
Launch with the model you have defined in [config/navigation.yaml](config/navigation.yaml)

#### With Gemini API
```bash
export GOOGLE_API_KEY=<your_key>
```

#### OR with local VLM inference

See [vlm/models.py](vlm/models.py) for a reference of the available model options, or to add a new model.

```bash
python vlm_server.py  --model llava-v1.6-mistral-7b-hf
```


### Navigation Demos

Run object-goal navigation in the Open3D simulator:

```bash
conda activate openfrontier
mkdir -p output/navigation
python demo.py \
  --mesh examples/mv2HUxq3B53.glb \
  --target toilet \
  --config config/navigation.yaml \
  --write_path output/navigation/navigation_state.json
```

This opens an observer view and a robot egocentric view. Use the robot window
for manual camera motion (`WASD` translation, `JL` rotation, `QZ` height), then
press `SPACE` in either window to start navigation. Add `--auto_start` to begin
immediately or `--vis_graph` to draw the topological graph.

Depth source can be selected with:

```bash
--depth_source GT
--depth_source Metric3D
--depth_source UniK3D
```

For a step-by-step exploration demo without object-goal scoring:

```bash
mkdir -p output/exploration
python demo.py \
  --mesh examples/mv2HUxq3B53.glb \
  --config config/exploration.yaml \
  --write_path output/exploration/exploration_state.json
```


## Habitat Benchmarking Pipeline

Use this pipeline for `benchmark.py` on HM3D, MP3D, and OVON. Start from the
Habitat-Lab/Habitat-Sim installation instead of creating the Python 3.11
`openfrontier` environment above.

First follow the [Habitat-Lab installation instructions for the <b>0.2.4 branch</b>](https://github.com/facebookresearch/habitat-lab/),
including its conda environment and Habitat-Sim install:

```bash
git clone https://github.com/facebookresearch/habitat-lab/
cd habitat-lab
git checkout v0.2.4
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines  # install habitat_baselines
```
Then install OpenFrontier inside that Habitat environment:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -r requirements_habitat.txt
bash scripts/download_weights.sh
```

### Datasets

```bash
mkdir data
```

Then follow the guidelines to download:

- [HM3D Val v0.2 Scenes](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d)

  - Rename scene_datasets/hm3d to scene_datasets/hm3d_v0.2

- [ObjectNav HM3D V2 dataset](https://github.com/facebookresearch/habitat-lab/blob/v0.2.4/DATASETS.md#:~:text=objectnav%5Fhm3d%5Fv2%2Ezip)

- [MP3D Val Scenes](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset)

- [ObjectNav MP3D V1 dataset](https://github.com/facebookresearch/habitat-lab/blob/v0.2.4/DATASETS.md#:~:text=objectnav%5Fhm3d%5Fv2%2Ezip)



Set dataset and Habitat roots before running Habitat benchmarks:

```bash
export OPENFRONTIER_DATA_ROOT= # e.g. /home/user/openfrontier/data
export HABITAT_LAB_ROOT= # e.g. /home/user/habitat-lab/habitat-lab
```

`OPENFRONTIER_DATA_ROOT` should contain `scene_datasets/hm3d_v0.2/val`, `scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json`
`objectnav_hm3d_v2/`, `objectnav_mp3d_v1/`, and/or `ovon/`. If your Habitat
ObjectNav YAMLs live outside `HABITAT_LAB_ROOT`, override them directly:

```bash
export HABITAT_HM3D_CONFIG=<objectnav_hm3d_yaml>
export HABITAT_MP3D_CONFIG=<objectnav_mp3d_yaml>
```

Expected dataset layout:

```bash
OPENFRONTIER_DATA_ROOT/
  scene_datasets/
    hm3d_v0.2/
    mp3d/
  objectnav_hm3d_v2/val/content/*.json.gz
  objectnav_mp3d_v1/val/content/*.json.gz
  ovon/val_unseen/content/*.json.gz
```

### Benchmarking

The benchmark runner supports `hm3d`, `mp3d`, and `ovon`:

```bash
python benchmark.py \
  --benchmark hm3d \
  --nickname openfrontier_hm3d \
  --config config/navigation.yaml \
  --output-path output
```

Run a small smoke test:

```bash
python benchmark.py \
  --benchmark hm3d \
  --nickname smoke_test \
  --config config/navigation.yaml \
  --output-path output \
  --eval_episodes 1 \
  --max_steps 500
```

Run MP3D or OVON:

```bash
python benchmark.py \
  --benchmark mp3d \
  --nickname openfrontier_mp3d \
  --config config/navigation.yaml \
  --output-path output

python benchmark.py \
  --benchmark ovon \
  --nickname openfrontier_ovon \
  --config config/navigation.yaml \
  --output-path output
```

Split a benchmark across workers:

```bash
python benchmark.py \
  --benchmark hm3d \
  --nickname openfrontier_hm3d \
  --config config/navigation.yaml \
  --output-path output \
  --split 1 4
```

Launch the same command with `--split 2 4`, `--split 3 4`, and `--split 4 4`
on other workers.

## Pipeline Configuration

YAML configs in `config/` control camera intrinsics, frontier detection,
mapping, planning, and VLM sources. Navigation configs include:

- `config/navigation.yaml`: default OpenFrontier navigation stack.
- `config/exploration.yaml`: Open3D exploration demo settings.

Important settings include `predict_interval`, `plan_interval`,
`segmentation_source`, `probabilities_source`, `detection_source`,
`success_threshold`, Wavemap mapping parameters, and OMPL planning parameters.

### Choosing VLMs

Set the VLMs in the navigation config file, for example
`config/navigation.yaml`:

```yaml
segmentation_source: sam3
probabilities_source: gemini-2.5-flash
detection_source: gemini-2.5-flash
```

`probabilities_source` scores which frontier is most likely to lead to the
target object. `detection_source` checks whether the target object is already
visible. `segmentation_source` controls object segmentation and is usually
`sam3`, though Gemini segmentation is also supported.

If `probabilities_source`, `detection_source`, or `segmentation_source` is a
Gemini model or a Gemma `*-api` model, OpenFrontier uses the configured Google
API key from `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

For local VLMs, set the config value to one of the local model names and run the
VLM server with the same model:

```yaml
probabilities_source: InternVL3_5-8B
detection_source: InternVL3_5-8B
```

```bash
python vlm_server.py --model InternVL3_5-8B
```

The local server listens on port `12185`; non-API VLM choices are routed there.
Supported model names are defined in `vlm/models.py` and include InternVL,
Gemma local variants, and LLaVA.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{openfrontier2026,
  title     = {OpenFrontier: General Navigation with Visual-Language Grounded Frontiers},
  author    = {Padilla-Cerdio, Esteban and Sun, Boyang and Pollefeys, Marc and Blum, Hermann},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026}
}
```
