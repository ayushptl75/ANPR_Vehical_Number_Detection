# ANPR Accuracy Evaluation Summary Report

## Overall Measured Performance Metrics

- **Total Categories Tested**: 12
- **Plate Detection Accuracy**: 100.0%
- **OCR Character Accuracy**: 0.0%
- **Full Plate Recognition Accuracy**: 0.0%
- **False Detection Rate**: 0.0%
- **Average Processing Speed**: 334.6 FPS
- **Average OCR Confidence**: 0.0%

## Category Breakdown Results

| Category | Ground Truth | Predicted Plate | Det. Acc (%) | OCR Acc (%) | Full Match (%) | FPS | Conf (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Clear front-facing plates | `MH12AB1234` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 334.2 | 0.0% |
| 2. Low-light plates | `DL1CG5692` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 328.6 | 0.0% |
| 3. Bright sunlight | `KA01MJ8821` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 387.7 | 0.0% |
| 4. Different distances | `TN07CB4510` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 440.1 | 0.0% |
| 5. Angled plates | `GJ01BC9012` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 192.9 | 0.0% |
| 6. Moving vehicles | `HR26DK3321` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 329.5 | 0.0% |
| 7. Small plates | `WB02AC1100` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 306.9 | 0.0% |
| 8. Multiple vehicles | `UP32EF9988` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 439.4 | 0.0% |
| 9. Different Indian state formats | `22BH1234A` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 259.6 | 0.0% |
| 10. Video input | `MH14DT4432` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 407.0 | 0.0% |
| 11. Live webcam | `RJ14CB7766` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 431.9 | 0.0% |
| 12. RTSP/IP camera | `AP09BD5544` | `NOT_DETECTED` | 100.0% | 0.0% | 0.0% | 384.7 | 0.0% |
