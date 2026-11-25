# DANC3D: 실시간 3차원 자세 추정 딥러닝 모델 기반 안무 교육 시스템

웹캠과 3D 아바타를 활용한 K-pop 댄스 학습·평가 서비스입니다.
사용자의 동작을 기준 안무와 비교해 실시간 점수와 부위별 피드백을 제공하고,
연습 결과를 저장해 곡별 기록과 최고 점수를 확인할 수 있습니다.

## 프로젝트 구조

```text
DANC3D/
├── app/                  # FastAPI 백엔드
│   ├── api/              # REST·WebSocket 라우터
│   ├── core/             # 자세 특징, DTW 시간 정렬, 점수·피드백
│   ├── services/         # 연습 세션, 인증, MediaPipe 모델 준비
│   └── db/               # SQLite 저장소
├── frontend/             # React + Vite 프론트엔드
│   ├── src/
│   │   ├── pages/        # 화면
│   │   ├── components/   # 공통 UI
│   │   ├── hooks/        # 카메라·연습 흐름
│   │   ├── services/     # API·평가 통신
│   │   └── retargeting/  # 3D 아바타 리타겟팅·발 접지 IK
│   └── public/           # 3D 캐릭터, 기준 아바타 데이터, 이미지
├── data/
│   ├── songs.json        # 곡 정보
│   └── references/       # 평가용 기준 동작 (NPZ)
├── preprocessing/        # AI Hub 원본으로 기준 데이터 생성
├── demo/                 # 데스크톱 데모
├── models/               # MediaPipe 모델 (첫 실행 시 자동 다운로드)
└── tests/                # 테스트
```

## 실행 방법

Python 3.10 이상, Node.js 22.12 이상이 필요합니다.

### 1. 원본 데이터 준비

AI Hub에서 [「K-pop 안무 영상」](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=52)을 신청해
아래 10곡을 받은 폴더 이름 그대로 `data/raw/`에 배치합니다. 정면 시점은 자동으로 선택됩니다.

```text
data/raw/
├── [원천]곡명 (가수)/   # 이미지
└── [라벨]곡명 (가수)/   # 라벨 JSON
```

아래는 이 프로젝트에서 사용하는 10곡으로, AI Hub에 표기된 이름 기준입니다.

| 곡 | 가수 |
|---|---|
| RingDingDong | 샤이니 |
| Dumb Dumb | 레드벨벳 |
| 영웅 | NCT127 |
| Gee | 소녀시대 |
| 소원을 말해봐 | 소녀시대 |
| Lion Heart | 소녀시대 |
| 빨간맛 | 레드벨벳 |
| 다시만난세계 | 소녀시대 |
| 쏘리쏘리 | 슈퍼주니어 |
| 러시안룰렛 | 레드벨벳 |

### 2. 기준 데이터 생성

```sh
python -m pip install -r requirements.txt

# 평가용 기준 동작 → data/references/
python preprocessing/prepare_reference_pose.py --all

# 기준 아바타 데이터 → frontend/public/references/
python preprocessing/export_reference.py
```

첫 실행 시 MediaPipe 모델을 자동으로 내려받으므로 인터넷 연결이 필요합니다.
평가용 기준 동작 생성은 약 3만 프레임을 처리하므로 10분 이상 걸릴 수 있습니다.

### 3. 실행

```sh
# 백엔드 (프로젝트 루트)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```sh
# 프론트엔드 (별도 터미널)
cd frontend
npm ci
npm run dev
```

http://127.0.0.1:5173 에 접속합니다.

## 출처

- **데이터:** [AI Hub 「K-pop 안무 영상」](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=52) (aihub.or.kr)
  이 프로젝트의 기준 안무 데이터는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원(NIA)의 지원을 받아 구축된 위 데이터를 가공하여 사용했습니다.
- **자세 추정 모델:** [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Google, Apache License 2.0)
- **3D 캐릭터:** [Adobe Mixamo](https://www.mixamo.com)
- **곡 썸네일:** [멜론](https://www.melon.com)의 앨범 커버 이미지이며, 저작권은 각 아티스트와 소속사에 있습니다.
