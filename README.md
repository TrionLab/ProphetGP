# ProphetGP

실험 CSV를 넣고, 다음에 어떤 조건을 시도할지 추천받는 도구입니다.

반응물(분자 이름 / SMILES / CAS)과 반응 조건을 학습한 뒤, Gaussian Process와 Bayesian Optimization으로 후보를 제안합니다.

---

## 무엇을 하나요

1. 지금까지의 실험 기록을 CSV로 읽습니다.
2. 분자 표현을 맞추고(필요하면 이름·CAS → SMILES), 숫자 특징으로 바꿉니다.
3. GP로 타깃 값과 불확실성을 함께 학습합니다.
4. 목표에 맞는 다음 실험 조건을 추천합니다.

다중 타깃(예: Emission Peak와 FWHM을 동시에)도 설정할 수 있습니다.

```mermaid
flowchart LR
    A[CSV] --> B[입력 정리]
    B --> C[분자 표현 통일]
    C --> D[피처 변환]
    D --> E[GP 학습]
    E --> F[BO 추천]
```

---

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

브라우저 UI까지 쓰려면:

```bash
pip install -e ".[dev,web]"
```

---

## 바로 써보기

샘플 데이터와 예시 설정이 포함되어 있습니다.

```bash
prophet-gp train --data data/sample/emission_experiments_demo.csv --config configs/example_open_reactants.yaml

prophet-gp suggest --data data/sample/emission_experiments_demo.csv --config configs/example_open_reactants.yaml --n-candidates 5
```

탐색을 더 넓히고 싶으면 `--strategy best_information`을 붙입니다. 기본은 `best_output`(성능 중심)입니다.

데이터 병합:

```bash
prophet-gp append --base-data data/raw/reactions.csv --new-data data/raw/new_batch.csv --out data/raw/reactions_merged.csv --config configs/default.yaml
```

---

## 샘플 데이터 (`data/sample`)

같은 실험 계열의 CSV입니다. 컬럼은 `Mol.1`, `Temperature`, `Emission Peak`, `FWHM`, `PLQY` 등입니다.

| 파일 | 설명 |
|------|------|
| `emission_experiments.csv` | 기본 학습용. PLQY에 결측이 일부 있습니다. |
| `emission_experiments_plqy_filled.csv` | 위와 같은 행 구성에 PLQY 결측을 채운 버전. |
| `emission_experiments_demo.csv` | 퀵스타트·데모용. 기본 세트에 실험 1건이 더 있습니다. |
| `emission_experiments_full.csv` | 가장 큰 세트. 타깃이 비어 있는 행도 포함합니다. |
| `new_batch_example.csv` | 퀵스타트에서 CSV 병합 예시에 쓰는 소량 배치. |

---

## 예시 설정 (`configs`)

| 파일 | 용도 |
|------|------|
| `default.yaml` | 컬럼명·옵션의 기본 골격. 새 설정을 만들 때 출발점으로 쓰면 됩니다. |
| `example_morgan_fp.yaml` | `morgan_fp` featuriser, Emission Peak / FWHM. |
| `example_topo_physchem.yaml` | `topo_physchem`, 반응물 허용 목록 포함. |
| `example_open_reactants.yaml` | 반응물 목록 제한 없이 탐색. 데모에 쓰입니다. |
| `example_limited_reactants.yaml` | 반응물 허용 목록으로 탐색 범위를 좁힌 예시. |
| `example_with_plqy.yaml` | Emission Peak / FWHM / PLQY 세 타깃. |
| `example_with_plqy_limited.yaml` | 세 타깃 + 반응물 허용 목록. |

---

## CSV 형식

반응물 컬럼에는 이름, SMILES, CAS를 넣을 수 있습니다. 여러 반응물은 `|`로 이어 씁니다.

| Mol.1 | Temperature | Emission Peak | FWHM |
|---|---:|---:|---:|
| 1,5-Diaminonaphthalene | 200 | 506 | 75 |

---

## 설정에서 자주 쓰는 항목

```yaml
data:
  reactant_column: Mol.1
  target_column:
    - "Emission Peak"
    - "FWHM"
  reactant_delimiter: "|"
  reactant_allowed_values: []   # 비우면 제한 없음
  condition_ranges:
    Temperature:
      min: 0
      max: 600
  ignore_columns:
    - PLQY
  explicit_condition_types:
    Temperature: continuous

featurization:
  featuriser: topo_physchem
  combine_strategy: concat

optimization:
  objective: target
  suggestion_strategy: best_output
  target_objectives:
    Emission Peak:
      objective: target
      target_value: 490.0
      weight: 1.0
    FWHM:
      objective: minimize
      weight: 1.0
  n_candidates: 3
```

- `objective`: `maximize` / `minimize` / `target`
- `suggestion_strategy`: `best_output`(성능) 또는 `best_information`(불확실성이 큰 쪽)
- `target_objectives`: 타깃마다 목표와 가중치를 다르게 줄 때 사용
- `reactant_allowed_values` / `condition_ranges`: 추천이 나갈 수 있는 범위를 제한

자세한 예시는 `configs/example_*.yaml`을 보면 됩니다.

---

## 추천 결과 읽는 법

`suggest` 출력에서 보면 될 필드:

- `mapped_reactants_input`, `Temperature` 등 — 실험에 그대로 쓸 조건
- `predicted_target_mean` / `predicted_target_std` — 예측값과 불확실성
- `target_gap` — `objective: target`일 때 목표와의 차이
- `objective_score` / `information_score` / `total_score` — 점수와 순위 근거
- `ranked_candidates` — `total_score` 기준 정렬 결과

---

## 브라우저 UI

```bash
pip install -e ".[web]"
prophet-gp-web
```

http://127.0.0.1:8765/ 에서 설정 편집, 학습, 추천, CSV 병합을 할 수 있습니다. 저장소 루트에서 실행하는 것이 좋습니다.

선택 환경 변수: `PROPHET_GP_ROOT`, `PROPHET_GP_WEB_HOST`, `PROPHET_GP_WEB_PORT`, `PROPHET_GP_WEB_RELOAD`.

학습 세션은 서버 메모리에만 있습니다. 서버를 재시작하면 다시 학습해야 합니다. HTML을 `file://`로만 열면 API가 동작하지 않습니다.

---

## 노트북

- `notebooks/prophetgp_quickstart.ipynb` — 학습·추천·예측·병합 흐름
- `notebooks/input_feature_walkthrough.ipynb` — 입력이 GP 벡터로 바뀌는 과정

셀은 위에서부터 순서대로 실행하세요. `notebooks/dev/` 아래 파일은 개발용이며 공개 튜토리얼이 아닙니다.

---

## 자주 겪는 일

**이름/CAS 변환이 실패할 때**  
PubChem 조회 실패나 표기 문제일 수 있습니다. SMILES를 직접 넣는 편이 안정적입니다.

**`InputDataWarning`이 나올 때**  
스케일 관련 권고인 경우가 많습니다. 실행 자체는 보통 계속됩니다. `standardize_gp_inputs` / `standardize_gp_targets`를 켜 보는 것도 방법입니다.

**추천이 숫자 벡터처럼만 보일 때**  
`decoded_candidates` 쪽의 해석 필드(`mapped_reactants_input`, 조건 컬럼, 점수)를 보면 됩니다.

---

## 프로젝트 구조

```text
configs/                 # 설정 YAML
data/sample/             # 공개용 샘플 CSV
notebooks/               # 공개 튜토리얼
  dev/                   # 개발용 스크래치 (공개 튜토리얼 아님)
src/prophet_gp/
  chem/                  # 분자 변환
  data/                  # 로더·스키마
  features/              # featuriser
  models/                # GP
  optimization/          # BO
  pipeline/              # 학습·추천 오케스트레이션
  web/                   # 브라우저 UI
  cli.py
tests/
```
