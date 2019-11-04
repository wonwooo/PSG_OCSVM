# PSG_OCSVM

OCSVM의 hyperparameter selection을 위한 PseudoSampleGeneration의 구현 코드와 실험결과 저장소입니다.

15개의 Benchmark dataset에 대한 실험은 jupyter lab에서 수행 되었습니다.

donut.csv는 간단한 활용례를 위한 2D 데이터입니다.
## 2-D 활용례

dount.csv와 PSG.py가 같은 프로젝트 디렉토리 내에 있어야 합니다.

import pandas as pd
import PSG

donut = pd.read_csv('donut.csv')
model_donut = PSG.PseudoSamples(donut, 2)
model_donut.search_optimal_hyperparameters()

