import base64
import cv2
import numpy as np
from app import app

img = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.putText(img, 'KA01AB1234', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
_, buf = cv2.imencode('.jpg', img)
data = base64.b64encode(buf).decode('ascii')
client = app.test_client()
resp = client.post('/api/live-detect', data={'frame_data': 'data:image/jpeg;base64,' + data})
print(resp.status_code)
print(resp.get_data(as_text=True))
