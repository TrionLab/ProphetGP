# ProphetGP

`ProphetGP`는 화학 실험 데이터를 보고,  
"다음에는 어떤 조건으로 실험하면 좋을까?"를 추천해주는 프로그램입니다.

쉽게 말해:

- 네가 지금까지 한 실험 기록(CSV)을 넣으면
- 컴퓨터가 패턴을 배우고
- 다음 실험 후보를 똑똑하게 골라줍니다.

---

## 1) 이 프로그램이 하는 일 (아주 쉽게)

요리를 예로 들어볼게요.

- 입력: 재료(분자), 오븐 온도(반응 조건), 시간(반응 조건)
- 출력: 맛 점수(물성값, target)

`ProphetGP`는 "어떤 재료 + 어떤 온도/시간이면 원하는 결과가 나올지"를 배우고,  
다음에 해볼 만한 조합을 추천해줍니다.

---

## 2) 핵심 기능

- 반응물 입력 지원: **분자 이름 / SMILES / CAS No.**
- 자동 변환: 이름/CAS를 가능한 경우 **SMILES**로 변환
- 반응물 여러 개 입력 가능 (`A|B|C` 형태)
- `gauche` featuriser 선택 가능 (동적 탐색)
- 반응 조건 타입 자동 추론:
  - `categorical` (종류형: 예, catalyst 이름)
  - `continuous` (연속형: 예, 온도)
  - `discrete` (정수형 단계값)
- Gaussian Process(GP) 모델 학습
- Bayesian Optimization(BO) 기반 후보 추천
- 다중 타깃 지원 (예: `Emission Peak`, `FWHM` 동시 최적화)
- 타깃별 objective/target_value/weight 개별 설정
- 입력 범위 제약 지원:
  - 반응물 허용 목록
  - 조건별 min/max
  - 조건별 allowed_values
- (선택) **브라우저 UI**: 설정 편집, 학습, 추천 결과 시각화(아래 **6) 브라우저 웹 UI**)

---

## 3) 동작 흐름 도식

```mermaid
flowchart LR
    A[CSV 실험 데이터] --> B[입력 정리]
    B --> C[분자 표현 통일<br/>name/CAS -> SMILES]
    C --> D[피처 변환<br/>gauche featuriser]
    D --> E[GP 학습]
    E --> F[BO 후보 생성]
    F --> G[입력 범위 제약 적용]
    G --> H[사람이 읽기 쉬운 추천 결과]
```

---

## 4) 설치 방법

터미널(CLI)만 쓸 때:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

**브라우저 웹 UI**까지 쓰려면 `web` 옵션을 함께 설치합니다.

```bash
pip install -e ".[dev,web]"
```

---

## 5) 가장 빠른 사용법

아래 예시는 저장소에 포함된 `configs/test.yaml`과 `data/sample/sample_data.csv`를 기준으로 합니다.  
다른 설정 파일 이름을 쓰는 경우 `--config`만 바꿔 주면 됩니다.

### 5-1. 학습

```bash
prophet-gp train --data data/sample/sample_data.csv --config configs/test.yaml
```

### 5-2. 추천

```bash
prophet-gp suggest --data data/sample/sample_data.csv --config configs/test.yaml --n-candidates 5
```

전략 선택도 가능:

```bash
prophet-gp suggest --data data/sample/sample_data.csv --config configs/test.yaml --n-candidates 5 --strategy best_information
```

### 5-3. 데이터 추가 병합

```bash
prophet-gp append --base-data data/raw/reactions.csv --new-data data/raw/new_batch.csv --out data/raw/reactions_merged.csv --config configs/default.yaml
```

---

## 5-4) 3분 튜토리얼 (그대로 복붙)

아래 순서대로 하면 샘플 데이터로 학습/추천까지 바로 확인할 수 있습니다.

### Step 1) 가상환경 + 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 2) 학습 실행

```bash
prophet-gp train --data data/sample/sample_data.csv --config configs/test.yaml
```

정상이라면 `trained`, `rows`, `features` 같은 정보가 나옵니다.

### Step 3) 추천 실행

```bash
prophet-gp suggest --data data/sample/sample_data.csv --config configs/test.yaml --n-candidates 5
```

정상이라면 아래 같은 정보가 JSON으로 나옵니다.

- `decoded_candidates`
- `predicted_target_mean`
- `predicted_target_std`
- `total_score`
- `ranked_candidates`

예상 출력 예시(일부):

```json
{
  "decoded_candidates": [
    {
      "predicted_target_mean": {
        "Emission Peak": 505.84,
        "FWHM": 95.78
      },
      "predicted_target_std": {
        "Emission Peak": 3.16,
        "FWHM": 3.25
      },
      "target_gap": {
        "Emission Peak": 10.84,
        "FWHM": null
      },
      "objective_score": -58.74,
      "information_score": 4.79,
      "total_score": -58.74,
      "ranking_strategy": "best_output",
      "mapped_reactants_input": "1,5-Diaminonaphthalene",
      "Temperature": 143.09
    }
  ],
  "ranked_candidates": [
    {
      "...": "total_score 높은 순으로 정렬된 후보"
    }
  ]
}
```

#### 출력 해석 방법 (중요)

- `predicted_target_mean`
  - 모델이 예상한 결과값입니다.
  - 예: `Emission Peak`가 505.84쯤 나올 것 같다는 뜻

- `predicted_target_std`
  - 모델의 불확실성(자신감 부족 정도)입니다.
  - 값이 클수록 "잘 모르는 영역"일 가능성이 큽니다.

- `target_gap`
  - `objective: target`일 때 목표값과의 차이입니다.
  - 작을수록 목표에 가깝습니다.
  - `minimize`/`maximize`만 쓸 때는 `null`일 수 있습니다.

- `objective_score`
  - 성능 기준 점수입니다.
  - 타깃별 objective와 weight를 반영한 합계입니다.

- `information_score`
  - 정보 획득 기준 점수입니다.
  - 불확실성이 큰 후보일수록 높아집니다.

- `total_score`
  - 실제 순위에 사용된 점수입니다.
  - `ranking_strategy=best_output`이면 `objective_score`
  - `ranking_strategy=best_information`이면 `information_score`

- `mapped_reactants_input`, `Temperature` 등
  - 사람이 바로 실험에 사용할 수 있도록 해석된 입력 조건입니다.

### Step 4) 탐색 전략 바꿔보기

```bash
prophet-gp suggest --data data/sample/sample_data.csv --config configs/test.yaml --n-candidates 5 --strategy best_information
```

`best_output`은 성능 중심, `best_information`은 정보 획득 중심 추천입니다.

---

## 6) 브라우저 웹 UI

FastAPI 기반 웹 서버와 단일 페이지(`index.html`)로, 노트북 `notebooks/prophetgp_quickstart.ipynb`와 같은 흐름을 브라우저에서 실행할 수 있습니다.

### 6-1. 설치

```bash
pip install -e ".[web]"
```

개발 도구까지 함께 쓰려면 `pip install -e ".[dev,web]"` 로 한 번에 설치하면 됩니다.

### 6-2. 서버 기동

저장소 **루트 디렉터리**에서 실행하는 것을 권장합니다 (`configs/`, `data/` 경로가 그 기준으로 잡힙니다).

**방법 A — 콘솔 스크립트**

```bash
prophet-gp-web
```

**방법 B — uvicorn 직접 실행**

```bash
uvicorn prophet_gp.web.app:app --host 127.0.0.1 --port 8765
```

기본 접속 주소: **http://127.0.0.1:8765/**

### 6-3. 환경 변수 (선택)

| 변수 | 설명 |
|------|------|
| `PROPHET_GP_ROOT` | 프로젝트 루트가 자동으로 맞지 않을 때(예: 패키지 설치 위치만 다른 경우) 절대 경로로 지정 |
| `PROPHET_GP_WEB_HOST` | 바인딩 호스트 (기본 `127.0.0.1`) |
| `PROPHET_GP_WEB_PORT` | 포트 (기본 `8765`) |
| `PROPHET_GP_WEB_RELOAD` | `true` / `1` / `yes` 이면 코드 변경 시 자동 재시작(개발용) |

### 6-4. 화면에서 할 수 있는 일

1. **`configs` YAML** — 목록에서 파일 선택, 내용 편집, 검증 후 디스크에 저장  
2. **Featuriser 목록** — 사용 가능한 featuriser 이름 조회  
3. **학습** — 선택한 설정 파일 + CSV 경로(프로젝트 루트 기준 상대 경로 또는 루트 아래 절대 경로)로 GP 학습. 성공 시 서버에 **세션**이 만들어집니다.  
4. **다음 실험 후보** — `n_candidates`, `best_output` / `best_information` 전략으로 추천. 후보마다 예측 평균·표준편차, 점수, 매핑된 반응물·조건 등을 표와 막대 그래프로 표시합니다.  
5. **데이터 병합(선택)** — 기존 CSV + 신규 CSV를 이어 붙여 지정 경로에 저장  

표시되는 **숫자는 소수점 셋째 자리**까지 반올림해 보여 줍니다.

### 6-5. 주의 사항

- HTML 파일을 **`file://`로만 열면** 브라우저 보안 때문에 API 호출이 되지 않습니다. 반드시 위 주소처럼 **서버를 띄운 뒤** 접속하세요.  
- 학습 세션은 **서버 메모리**에만 있으며, 최대 32개까지 유지됩니다. 서버를 재시작하거나 세션이 밀려 나가면 **다시 학습**해야 추천을 이어갈 수 있습니다.  
- CSV·병합 출력 경로는 **프로젝트 루트 안**으로만 허용됩니다.

### 6-6. HTTP API (참고)

자동화나 외부 도구 연동 시 같은 프로세스에서 다음 엔드포인트를 사용할 수 있습니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 동작 확인 및 인식된 프로젝트 루트 |
| GET | `/api/configs` | `configs/` 아래 yaml 목록 |
| GET | `/api/config/{파일명}` | YAML 텍스트 및 검증 요약 |
| PUT | `/api/config/{파일명}` | YAML 저장(본문 UTF-8 텍스트, Pydantic 검증 통과 필요) |
| GET | `/api/featurisers` | featuriser 이름 목록 |
| POST | `/api/train` | JSON: `config_name`, `data_path` → `session_id` 등 |
| POST | `/api/suggest` | JSON: `session_id`, 선택적 `n_candidates`, `strategy` |
| POST | `/api/append` | JSON: `config_name`, `base_data_path`, `new_data_path`, `out_path` |

---

## 7) CSV 데이터 형식

### 단일 타깃 예시

| Mol.1 | Temperature | Emission Peak |
|---|---:|---:|
| 1,5-Diaminonaphthalene | 200 | 506 |

### 다중 타깃 예시

| Mol.1 | Temperature | Emission Peak | FWHM |
|---|---:|---:|---:|
| 1,5-Diaminonaphthalene | 200 | 506 | 75 |

> 반응물이 여러 개면 `Mol.1` 컬럼 값에 `|`로 이어서 넣습니다.  
> 예: `ethanol|CC(=O)O|64-17-5`

---

## 8) config 설정 가이드 (초등학생도 이해 가능 버전)

config는 "게임 옵션 창" 같은 것입니다.  
무엇을 입력으로 보고, 어떤 목표로 추천할지 정합니다.

### 전체 예시 (일러스트 — 저장소 실제 파일은 `configs/test.yaml`, `configs/default.yaml` 등을 참고)

```yaml
data:
  reactant_column: Mol.1
  target_column:
    - "Emission Peak"
    - "FWHM"
  reactant_delimiter: "|"

  reactant_allowed_values:
    - "1,5-Diaminonaphthalene"
    - "1,8-Diaminonaphthalene"
    - "2,3-Diaminonaphthalene"
    - "2,6-Diaminonaphthalene"

  condition_ranges:
    Temperature:
      min: 0
      max: 600
    catalyst:
      allowed_values: ["A", "B"]

  ignore_columns:
    - PLQY

  explicit_condition_types:
    Temperature: continuous

featurization:
  featuriser: ecfp_fingerprints
  combine_strategy: concat

optimization:
  objective: target
  suggestion_strategy: best_output
  target_value: 495.0

  target_objectives:
    Emission Peak:
      objective: target
      target_value: 495.0
      weight: 1.0
    FWHM:
      objective: minimize
      weight: 0.5

  n_restarts: 10
  raw_samples: 128
  target_search_size: 5000
  n_candidates: 5
```

### 항목별 설명

#### `data`

- `reactant_column`: 반응물 컬럼 이름
- `target_column`: 맞추고 싶은 결과 컬럼
  - 한 개면 문자열
  - 여러 개면 리스트
- `reactant_delimiter`: 반응물 여러 개를 나눌 문자
- `reactant_allowed_values`: 추천할 반응물 후보 제한 목록
  - 비우면 아무 반응물이나 가능(열린 범위)
- `condition_ranges`: 조건값 범위 제한
  - `min`/`max`: 숫자 범위
  - `allowed_values`: 허용 목록(주로 categorical)
- `ignore_columns`: 학습 입력에서 빼고 싶은 컬럼
- `explicit_condition_types`: 타입을 강제로 지정

#### `featurization`

- `featuriser`: 분자를 숫자로 바꾸는 방법
  - 예: `ecfp_fingerprints`, `molecular_graphs`
- `combine_strategy`: 현재 `concat` 사용

#### `optimization`

- `objective`: 기본 목표 (`maximize`, `minimize`, `target`)
- `suggestion_strategy`:
  - `best_output`: 성능이 좋아 보이는 점 추천
  - `best_information`: 정보가 많이 늘어날 점 추천(탐색)
- `target_value`: `objective: target`일 때 기준값
- `target_objectives`: 다중 타깃일 때 타깃별 옵션
  - 각 타깃마다 `objective`, `target_value`, `weight` 가능
- `n_candidates`: 추천 개수
- `target_search_size`: 후보 풀 샘플 크기(클수록 탐색 넓음)

---

## 9) 용어 사전 (아주 쉬운 말)

- **SMILES**: 분자를 글자로 적는 방법
- **featuriser**: 글자/구조를 숫자 벡터로 바꾸는 도구
- **GP (Gaussian Process)**: "값 예측 + 불확실성"을 같이 주는 모델
- **BO (Bayesian Optimization)**: 실험 횟수를 아끼며 좋은 조건을 찾는 전략
- **target objective**:
  - `maximize`: 크게 만들기
  - `minimize`: 작게 만들기
  - `target`: 특정 값에 맞추기
- **weight**: 타깃 중요도 점수

---

## 10) 출력 결과 읽는 방법

`suggest` 결과의 주요 필드:

- `predicted_target_mean`: 타깃 예측값
- `predicted_target_std`: 예측 불확실성
- `target_gap`: 목표값과의 차이(해당 시)
- `objective_score`: 타깃 목적함수 기준 점수
- `information_score`: 정보획득 점수
- `total_score`: 실제 순위 계산에 사용된 점수
- `mapped_reactants_input`: 사람이 읽기 쉬운 반응물 입력

---

## 11) 자주 만나는 질문

### Q1. 이름/CAS를 넣었는데 실패해요.
- 인터넷/PubChem 조회 실패, 오타, 특수한 명명 문제일 수 있습니다.
- 가능하면 SMILES를 직접 넣으면 가장 안정적입니다.

### Q2. 경고(`InputDataWarning`)가 떠요.
- 보통 스케일링 권고 경고입니다.
- 실행 실패는 아니며, 데이터가 커질수록 전처리/정규화 튜닝이 도움이 됩니다.

### Q3. 추천 결과가 raw 벡터라 읽기 어려워요.
- 현재는 `decoded_candidates`에 해석 필드가 같이 제공됩니다.
- `mapped_reactants_input`, 조건 컬럼 값, 점수 필드를 같이 보세요.

---

## 12) 프로젝트 구조

```text
configs/                  # 설정 파일
data/
  raw/                    # 원본/누적 데이터셋
  sample/                 # 샘플 CSV
  processed/              # 전처리 결과 캐시
notebooks/                # 사용 예시 노트북
scripts/                  # 실행 유틸
src/prophet_gp/
  chem/                   # 분자 변환/검증
  data/                   # 데이터 로더/스키마
  features/               # gauche featuriser 어댑터
  models/                 # GP surrogate
  optimization/           # BO 로직
  pipeline/               # 전체 오케스트레이션
  web/                    # 브라우저 UI (FastAPI + static/index.html)
    static/
      index.html
  cli.py                  # CLI 진입점
tests/                    # 테스트
```