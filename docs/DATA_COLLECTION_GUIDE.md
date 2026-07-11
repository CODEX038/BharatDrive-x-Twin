# DATA_COLLECTION_GUIDE

All recording with informed consent, anonymous participant IDs (P01, P02…), parked vehicle / classroom / simulator only. Never record a genuinely fatigued person driving on a public road.

## 1. Driver-camera clips

Target: 3–5 participants × 10–15 scenarios × 30–60 s. Camera at dashboard height, face fills ~1/4 of frame, 720p, 25–30 fps.

Scenarios per participant (file naming `P01_alert_day.mp4` etc.):
alert face · normal blinking · slow blinking · long blink · repeated eye closure · yawning · speaking · laughing · drinking · looking left/right/down · head nodding · phone distraction · face missing · partial occlusion · spectacles · sunglasses · low light · backlighting · camera vibration · alternate camera position.

Label each clip in `datasets/registry.json` with participant ID, scenario, lighting, eyewear. This volume supports calibration and controlled testing only — not universal performance claims.

## 2. Front-road clips

30–50 clips × 20–60 s, camera rigidly mounted (windscreen/dashboard), recorded as passenger or from legally obtained footage. Scenes: normal urban, highway, rural, narrow, market, school zone, two-wheeler traffic, auto-rickshaw traffic, pedestrian crossing, wrong-side vehicle, pothole, speed breaker, animal crossing, construction, waterlogging, night, rain, poor visibility, parked truck, broken signal.

## 3. CCTV-style footage

Only: prerecorded authorized junction footage, college CCTV with written permission, user-owned camera, public legal research footage, or SUMO-rendered simulated views. Scenes: normal junction, heavy traffic, long queue, wrong-side movement, crowd, blockage, waterlogging, broken signal, construction, accident-like stopped traffic.

## 4. Photos (custom Indian hazard classes)

100–200 varied images per class to start: pothole, speed breaker, auto-rickshaw, cattle, dog, waterlogging, open manhole, debris, construction barrier, wrong-side vehicle. Vary distance, angle, day/night, weather, occlusion, device. Annotation quality > quantity. Use CVAT or LabelImg; YOLO txt format; store under `datasets/custom_indian_hazards/`.

## 5. Pilot route packet

Pick one 2–5 km route. Provide: start/end points + approximate GPS, junction list, schools/markets/hospitals/bus stops on route, known speed breakers/potholes/blind turns/narrow sections/accident-prone spots, typical traffic pattern, road width/lanes, common road users. Photos and one drive-through video (as passenger). This feeds `digital_twin/osm_import.py`.
