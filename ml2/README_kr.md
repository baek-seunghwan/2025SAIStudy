# 2025 USW AI 스터디 대회 ML2 – 보험 사기 탐지

이 저장소는 **보험 사기(fraud) 분류** 문제에 대해 전처리 → 피처 엔지니어링 → 모델 학습/튜닝 → **임계값(Threshold) 최적화** → 제출(Submission)까지 이어지는 **E2E 파이프라인**을 담고 있습니다.

## 빠른 시작
```bash
pip install -r requirements.txt
# data/에 train.csv, test.csv, sample_submission.csv를 둡니다.
make train      # CV 학습 & OOF 저장
make search     # 임계값/양성수 그리드 탐색
make infer      # 최종 추론 & 제출 파일 생성
```

## 폴더 구조
```
.
├─ data/                # (비공개/미커밋) 원본 CSV 3종
├─ notebooks/           # EDA/SHAP 노트북
├─ src/                 # 파이프라인 코드
│  ├─ features.py
│  ├─ run_train.py
│  ├─ run_infer.py
│  └─ threshold_search.py
├─ submissions/         # 제출 파일 및 탐색 로그
├─ docs/                # 튜토리얼/결과/슬라이드
├─ config.yaml          # 하이퍼파라미터/경로
├─ requirements.txt
└─ .gitignore
```

## 피처 엔지니어링(예시)
- `month_num`, `claim_day_of_week_num`
- `payout_income_ratio = claim_est_payout / annual_income`
- `driver_vehicle_age_ratio/diff`
- `liab_payout = liab_prct * claim_est_payout`

## 임계값 최적화 전략
- **positive_quota**: 예측확률 상위 *N*개를 양성으로 설정(양성수 그리드 탐색)
- **max_f1**: OOF에서 macro F1이 최대가 되는 임계값 탐색

## 재현성
- `config.yaml`의 `seed`, `cv` 파라미터로 재현성 제어
- 제출 파일은 `submissions/`에 `submission_*.csv`로 저장

---

### 발표자료
`docs/slides_ML2_백승환_23017112_알고리즘분석.pptx` 포함 (10장)
