# NOWL — Traffic Event Detection & Accident Anticipation

## O'rnatish va ishga tushirish
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json

Vaznlar `weights/yolo11n.pt` da (repo ichida, ~5.6 MB).

## Yondashuv
YOLO11n (Ultralytics, COCO pretrained) + ByteTrack orqali obyekt detektsiya va kuzatuv.
Ustiga qurilgan qoida-asoslangan mantiq (src/events/) 6 ta hodisa sinfini aniqlaydi:
stopped_vehicle, wrong_way, jaywalking, congestion, failure_to_yield, collision_risk.
RiskEstimator time-to-collision (TTC) asosida ishlaydi.

Fine-tuning qilinmagan — barcha aniqlash mantiqi rule-based.

## Datasets/litsenziyalar
- COCO (YOLO11n pretrained vazn) — CC BY 4.0 / AGPL-3.0 (Ultralytics)
- ByteTrack — MIT

## Seed
Deterministik (YOLO inference seed=0 sifatida ishlatiladi, qo'shimcha randomness yo'q).

## Jamoa
- Anvar Mexmonov — AI & Pipeline Engineer (model, qoidalar, RiskEstimator)
- Sirojiddin Abduraxmonov — Data, Evaluation & Quality Engineer (dev-set, evaluate.py, sifat nazorati)
- Sardor Omonov — Full-Stack Web & Demo Engineer (sayt, EDA, Live Demo, Natijalar)