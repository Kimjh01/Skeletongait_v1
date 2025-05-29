# SkeletonGait: Gait Recognition Using Skeleton Maps

This [paper](https://arxiv.org/abs/2311.13444) has been accepted by AAAI 2024.

## 히트맵 및 훈련 단계 생성 (단일 학습 기준)

### Step 1: 히트맵 생성

단일 GPU 환경에서 히트맵을 생성하려면 아래 스크립트를 사용하세요:

```
CUDA_VISIBLE_DEVICES=0 \
python datasets/pretreatment_heatmap.py \
--pose_data_path=<your pose .pkl files path> \
--save_root=<your_path> \
--dataset_name=<dataset_name>
```

**Parameter Guide:**

* `--pose_data_path`: `.pkl` 형식의 포즈 데이터(ID 기준)가 저장된 디렉토리 (필수).
* `--save_root`: 생성된 히트맵(`.pkl`, ID 기준)을 저장할 루트 디렉토리 (필수).
* `--dataset_name`: 전처리할 데이터셋 이름 (필수).
* `--ext_name`: 저장 디렉토리에 붙일 선택적 접미사 (기본값: 빈 문자열).
* `--heatmap_cfg_path`: 히트맵 생성기 설정 파일 경로 (기본값: `configs/skeletongait/pretreatment_heatmap.yaml`).

**Note:** 포즈 데이터가 COCO 18 포맷(OpenPose 등 사용 시)이라면, 설정 파일(`configs/skeletongait/pretreatment_heatmap.yaml`)에서 `transfer_to_coco17=True`로 설정하세요.

---

### Step 2: 히트맵 및 실루엣 데이터 심볼릭 링크 생성 (선택 사항)

히트맵과 실루엣을 하나의 폴더로 연결하고 싶을 때는 다음 명령어를 사용하세요:

```
python datasets/ln_sil_heatmap.py \
--heatmap_data_path=<path_to_your_heatmap_folder> \
--silhouette_data_path=<path_to_your_silhouette_folder> \
--output_path=<path_to_your_output_folder>
```

**Parameter Guide:**

* `--heatmap_data_path`: 히트맵 데이터의 절대 경로 (필수).
* `--silhouette_data_path`: 실루엣 데이터의 절대 경로 (필수).
* `--output_path`: 결과 심볼릭 링크 폴더 경로 (필수).
* `--dataset_pkl_ext_name`: 실루엣 `.pkl` 확장명 설정 (선택 사항, 기본값: `.pkl`).

---

### Step 3: SkeletonGait 또는 SkeletonGait++ 훈련 (단일 학습 기준)

#### SkeletonGait 훈련

```
CUDA_VISIBLE_DEVICES=0 \
python opengait/main.py \
--cfgs ./configs/skeletongait/skeletongait_Gait3D.yaml \
--phase train --log_to_file
```

#### SkeletonGait++ 훈련

```
CUDA_VISIBLE_DEVICES=0 \
python opengait/main.py \
--cfgs ./configs/skeletongait/skeletongait++_Gait3D.yaml \
--phase train --log_to_file
```

---

## SkeletonGait 및 SkeletonGait++ 성능 비교

### SkeletonGait

| Datasets            | `Rank1`                                                                                                                                    | Configuration                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| CCPG                | CL: 52.4, UP: 65.4, DN: 72.8, BG: 80.9                                                                                                     | [skeletongait\_CCPG.yaml](./skeletongait_CCPG.yaml)           |
| OU-MVLP (AlphaPose) | TODO                                                                                                                                       | [skeletongait\_OUMVLP.yaml](./skeletongait_OUMVLP.yaml)       |
| SUSTech-1K          | Normal: 54.2, Bag: 51.7, Clothing: 21.34, Carrying: 51.59, Umberalla: 44.5, Uniform: 53.37, Occlusion: 67.07, Night: 44.15, Overall: 51.46 | [skeletongait\_SUSTech1K.yaml](./skeletongait_SUSTech1K.yaml) |
| Gait3D              | 38.1                                                                                                                                       | [skeletongait\_Gait3D.yaml](./skeletongait_Gait3D.yaml)       |
| GREW                | TODO                                                                                                                                       | [skeletongait\_GREW.yaml](./skeletongait_GREW.yaml)           |

### SkeletonGait++

| Datasets   | `Rank1`                                                                                                                                       | Configuration                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| CCPG       | CL: 90.1, UP: 95.0, DN: 92.9, BG: 97.0                                                                                                        | [skeletongait++\_CCPG.yaml](./skeletongait++_CCPG.yaml)           |
| SUSTech-1K | Normal: 85.09, Bag: 82.90, Clothing: 46.53, Carrying: 81.88, Umberalla: 80.76, Uniform: 82.50, Occlusion: 86.16, Night: 47.48, Overall: 81.33 | [skeletongait++\_SUSTech1K.yaml](./skeletongait++_SUSTech1K.yaml) |
| Gait3D     | 77.40                                                                                                                                         | [skeletongait++\_Gait3D.yaml](./skeletongait++_Gait3D.yaml)       |
| GREW       | 87.04                                                                                                                                         | [skeletongait++\_GREW.yaml](./skeletongait++_GREW.yaml)           |

