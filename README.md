# ProphetGP

`gauche` 기반 반응 물성 예측 + Gaussian Process Bayesian Optimization 프로젝트입니다.

## 주요 기능

- 반응 전 물질 입력 지원: 분자 이름 / SMILES / CAS No.
- 분자 표기 자동 정규화: 이름/CAS를 SMILES로 변환 (PubChem API + RDKit 검증).
- 다중 반응물 처리: 반응물 1개 이상 입력 가능.
- `gauche` featuriser 래퍼: 사용 가능한 모든 featuriser를 동적으로 조회/선택.
- 반응 조건 자동 타입 추론: categorical / continuous / discrete.
- config로 조건 타입 강제 지정 가능.
- Gaussian Process 기반 surrogate 모델 학습.
- Bayesian Optimization으로 다음 최적 실험 조건 추천.
- 데이터셋 증분 추가(append) 및 재학습 지원.

## 빠른 시작

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 데이터 준비

기본 CSV 필드 예시:

- `reactants`: 반응물 리스트 (`|` 구분, 예: `ethanol|CC(=O)O|64-17-5`)
- `target`: 예측/최적화할 물성값
- `temperature`, `time`, `catalyst` 등 반응 조건 컬럼

샘플 실행:

```bash
prophet-gp train --data data/raw/reactions.csv --config configs/default.yaml
prophet-gp suggest --data data/raw/reactions.csv --config configs/default.yaml --n-candidates 5
prophet-gp append --base-data data/raw/reactions.csv --new-data data/raw/new_batch.csv --out data/raw/reactions_merged.csv
```

## 프로젝트 구조

```text
configs/                  # 파이프라인 설정
data/
  raw/                    # 원본/누적 데이터셋
  processed/              # 전처리 결과 캐시
scripts/                  # 실행 유틸 스크립트
src/prophet_gp/
  chem/                   # 분자 표기 변환/검증
  data/                   # 스키마, 로더, 전처리
  features/               # gauche featuriser 통합
  models/                 # GP surrogate 모델
  optimization/           # Bayesian optimization
  pipeline/               # 상위 오케스트레이션
  cli.py                  # CLI 진입점
tests/
```