# AutoCount-AI

### Using YOLO v8 model to detect the count of moving objects on highway or elevators video.
----
#### Stack Used:
1. YOLO v8
2. opencv
3. cv2
4. SORT (simple online and realtime tracking)
----
#### Steps to get started:
1. setup env and install requirements
```
python3.10 -m venv env
source env/bin/activate # for linuxx/maxos

# installing requirements
pip install -r requirements.txt
```
2. run file(car-counter.py / people-counter.py)

```
python car-counter.py
python people-counter.py
```
----
#### Acknowledgement

Tracking powered by [SORT](https://github.com/abewley/sort) by @abewley.

