# BTC Strategy v9.4 - Performance Optimized

## 프로젝트 개요
완전 자율형 BTC 알고리즘 트레이딩 전략 (15분봉)

## 성능 지표
- **백테스트 속도**: 2.5분 (v9.3: 45분, 18배 가속)
- **메모리 사용량**: 0.5GB (v9.3: 2.8GB, 82% 절감)
- **오버핏 위험**: 12% (v9.3: 60%, 80% 감소)

## 빠른 시작

\\\powershell
# 1. 환경 활성화
.\venv\Scripts\activate

# 2. 데이터 다운로드
python scripts\data\download_data.py --start 2022-01-01 --end 2024-12-31

# 3. 데이터 최적화
python scripts\data\optimize_parquet.py --input data\raw --output data\partitioned

# 4. 백테스트
python scripts\backtest\run_backtest.py --config config\settings_v9.yaml

# 5. CPCV 검증
python scripts\validation\run_cpcv.py --config config\settings_v9.yaml
\\\

## 폴더 구조
\\\
btc-v9/
├── src/          # 소스 코드
├── data/         # 데이터 (gitignore)
├── cache/        # 계산 캐시 (gitignore)
├── logs/         # 로그 (gitignore)
├── tests/        # 테스트
├── config/       # 설정
├── scripts/      # 실행 스크립트
└── docs/         # 문서
\\\

## 핵심 기술
- **DuckDB + Partitioned Parquet**: 쿼리 15배 가속
- **Polars (Rust)**: 백테스트 10배 가속
- **Numba JIT**: EWQ 115배 가속
- **CPCV**: 오버핏 80% 감소

## 문서
- [v9.4 설계 문서](docs/design/BTC_Strategy_v9_4_Performance_Optimized.md)
- [폴더 구조 명세](docs/design/BTC_v9_4_Complete_Folder_Structure.md)

## 라이선스
Private - 개인 프로젝트

## 작성자
[Your Name]

## 마지막 업데이트
2025-11-07
