"""Main Flask application entry point for the ANPR system."""
from __future__ import annotations

import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for, send_from_directory, send_file
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from config import Config
from modules.alert_manager import AlertManager
from modules.camera import CameraManager
from modules.database_manager import DatabaseManager
from modules.live_processor import LiveProcessor
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.report_generator import ReportGenerator
from modules.utils import (
    compute_toll_amount,
    ensure_directories,
    get_safe_filename,
    hash_password,
    is_valid_plate_text,
    verify_password,
)
from modules.vehicle_detector import VehicleDetector
from modules.video_processor import VideoProcessor
from services.rto_vehicle_service import RTOVehicleService
from services.carinfo_service import CarInfoService
from modules.rto_worker import enqueue_lookup, get_job_status



def _encode_frame_data_uri(frame: np.ndarray) -> str:
    """Encode an OpenCV frame to a JPEG data URI."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return ''
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}"


def _save_detection_images(vehicle_image: Any, plate_image: Any) -> tuple[str, str]:
    """Save cropped detection images for storage and return file paths."""
    vehicle_path = ''
    plate_path = ''
    if vehicle_image is not None and hasattr(vehicle_image, 'shape') and vehicle_image.size != 0:
        vehicle_path = video_processor._save_image(vehicle_image, 'vehicle')
    if plate_image is not None and hasattr(plate_image, 'shape') and plate_image.size != 0:
        plate_path = video_processor._save_image(plate_image, 'plate')
    return vehicle_path, plate_path

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config.from_object(Config)

ensure_directories()

from modules.anpr_pipeline import ANPRPipeline

camera_manager = CameraManager()
vehicle_detector = VehicleDetector()
plate_detector = PlateDetector()
ocr_reader = OCRReader()
anpr_pipeline = ANPRPipeline(vehicle_detector, plate_detector, ocr_reader)
video_processor = VideoProcessor(vehicle_detector, plate_detector, ocr_reader)
live_processor = LiveProcessor(vehicle_detector, plate_detector, ocr_reader)
alert_manager = AlertManager()
report_generator = ReportGenerator()
database_manager = DatabaseManager()
rto_vehicle_service = RTOVehicleService(database_manager)
carinfo_service = CarInfoService(database_manager)

try:
    from routes.vehicle_routes import vehicle_bp
    app.register_blueprint(vehicle_bp)
except Exception:
    # blueprint optional
    pass


@app.before_request
def require_login() -> None:
    """Protect routes that require authentication."""
    public_routes = {"/login", "/logout", "/static/<path:filename>", "/api/health", "/api/spec"}
    if request.endpoint in {"static"}:
        return
    if request.path.startswith("/static/"):
        return
    if request.path in public_routes:
        return
    if "user_id" not in session:
        if request.path != "/login":
            return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    """Render the login page and authenticate administrators."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = database_manager.authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login successful", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials. Please use admin / admin123", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout() -> Any:
    """Clear current session and return to the login page."""
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("login"))


@app.route("/")
@app.route("/dashboard")
def dashboard() -> Any:
    """Render the main dashboard with summary statistics."""
    stats = database_manager.get_dashboard_stats()
    recent = database_manager.get_recent_detections(limit=8)
    alerts = alert_manager.get_alerts()
    camera_status = camera_manager.get_status()
    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        alerts=alerts,
        camera_status=camera_status,
    )


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image() -> Any:
    """Upload and process a still image for plate recognition."""
    if request.method == "POST":
        if "image" not in request.files:
            flash("No image file uploaded", "danger")
            return redirect(url_for("upload_image"))
        file = request.files["image"]
        if file.filename == "":
            flash("No image selected", "danger")
            return redirect(url_for("upload_image"))
        allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        if not file.filename.lower().endswith(allowed_ext):
            flash("Unsupported image format", "danger")
            return redirect(url_for("upload_image"))
        # verify image content
        try:
            img = Image.open(file.stream)
            img.verify()
            file.stream.seek(0)
        except UnidentifiedImageError:
            flash("Invalid or corrupted image file", "danger")
            return redirect(url_for("upload_image"))
        except Exception:
            flash("Unable to read uploaded image", "danger")
            return redirect(url_for("upload_image"))

        try:
            filename = get_safe_filename(file.filename)
            upload_path = Path(Config.UPLOAD_FOLDER) / filename
            file.save(upload_path)

            # Read image array and run unified ANPR core detection pipeline
            img_arr = cv2.imread(str(upload_path))
            result = anpr_pipeline.process_image(img_arr)

            plate = (result.get("plate_number") or result.get("plate") or "").strip()
            if plate == "NOT_DETECTED":
                plate = ""
            detected_vehicle_type = result.get("detected_vehicle_type") or "Unknown"
            detected_vehicle_confidence = float(result.get("detected_vehicle_confidence") or 0.0)
            carinfo_res = None
            verification_status = "DETECTED"
            vehicle_details = None

            if plate:
                carinfo_res = carinfo_service.get_vehicle_information(
                    plate,
                    ocr_confidence=float(result.get("confidence") or 1.0),
                    plate_confidence=detected_vehicle_confidence,
                )
                verification_status = carinfo_res.get("status", "DETECTED")
                if carinfo_res.get("verified"):
                    flash(f"Vehicle verified with authorized CarInfo API: {carinfo_res.get('manufacturer', '')} {carinfo_res.get('model', '')}", "success")
                elif verification_status == "LOW CONFIDENCE":
                    flash("Plate recognition confidence too low. API lookup skipped.", "warning")
                elif verification_status == "INVALID FORMAT":
                    flash("Invalid registration number format detected.", "warning")
                elif verification_status == "API ERROR":
                    flash(f"Vehicle detected. CarInfo API lookup notice: {carinfo_res.get('error', 'Service unavailable')}", "info")

                vehicle_details = database_manager.get_vehicle_info(plate) or carinfo_res


            # Build browser-accessible URLs for saved images
            original_img_url = url_for("serve_upload", filename=filename)
            vehicle_img_url = None
            plate_img_url = None
            processed_img_url = None

            vehicle_crop = result.get("vehicle_crop")
            plate_crop = result.get("plate_crop")
            annotated_frame = result.get("annotated_frame")

            if vehicle_crop is not None and hasattr(vehicle_crop, "shape") and vehicle_crop.size != 0:
                v_file = get_safe_filename(f"vehicle_{filename}")
                cv2.imwrite(str(Path(Config.VEHICLE_IMAGES_FOLDER) / v_file), vehicle_crop)
                vehicle_img_url = url_for("serve_vehicle_image", filename=v_file)

            if plate_crop is not None and hasattr(plate_crop, "shape") and plate_crop.size != 0:
                p_file = get_safe_filename(f"plate_{filename}")
                cv2.imwrite(str(Path(Config.PLATE_IMAGES_FOLDER) / p_file), plate_crop)
                plate_img_url = url_for("serve_plate_image", filename=p_file)

            if annotated_frame is not None and hasattr(annotated_frame, "shape") and annotated_frame.size != 0:
                proc_file = get_safe_filename(f"proc_{filename}")
                cv2.imwrite(str(Path(Config.PROCESSED_FOLDER) / proc_file), annotated_frame)
                processed_img_url = url_for("serve_processed", filename=proc_file)

            details = {
                **result,
                "detected_vehicle_type": detected_vehicle_type,
                "detected_vehicle_confidence": detected_vehicle_confidence,
                "plate_valid": carinfo_res.get("verified", False) if carinfo_res else False,
                "verification_status": verification_status,
                "carinfo": carinfo_res,
                "rto_status": verification_status,
                "rto_error": carinfo_res.get("error") if carinfo_res else None,
                "vehicle_details": vehicle_details or carinfo_res,
                "vehicle_image_url": vehicle_img_url,
                "plate_image_url": plate_img_url,
                "processed_image_url": processed_img_url,
                "original_image_url": original_img_url,
            }

            # Save detection only if not duplicate in short window
            if plate and database_manager.can_save_detection(plate, Config.MAX_DUPLICATE_SECONDS):
                database_manager.add_detection(
                    plate_number=plate,
                    vehicle_type=detected_vehicle_type,
                    confidence=float(result.get("confidence") or 0.0),
                    camera_name="Upload",
                    vehicle_image=vehicle_img_url or "",
                    plate_image=plate_img_url or "",
                    video_name=filename,
                    source_type="image_upload",
                    original_image=original_img_url or "",
                    processed_image=processed_img_url or "",
                    cropped_plate=plate_img_url or "",
                    lookup_status=verification_status,
                    lookup_error=carinfo_res.get("error") if carinfo_res else "",
                )


            flash(f"Processed image. Plate: {plate or 'Not detected'}", "success")
            return render_template("upload.html", result=details)
        except Exception as exc:
            app.logger.error("Error processing uploaded image: %s", exc, exc_info=True)
            flash(f"Error processing image: {str(exc)}", "danger")
            return render_template("upload.html")
    return render_template("upload.html")


@app.route("/upload-video", methods=["GET", "POST"])
def upload_video() -> Any:
    """Upload a video for frame-by-frame analysis and export."""
    if request.method == "POST":
        if "video" not in request.files:
            flash("No video file uploaded", "danger")
            return redirect(url_for("upload_video"))
        file = request.files["video"]
        if file.filename == "":
            flash("No video selected", "danger")
            return redirect(url_for("upload_video"))
        if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
            flash("Unsupported video format", "danger")
            return redirect(url_for("upload_video"))
        filename = get_safe_filename(file.filename)
        upload_path = Path(Config.UPLOAD_FOLDER) / filename
        file.save(upload_path)

        result = video_processor.process_video(str(upload_path))
        output_path = result.get("output_path") if isinstance(result, dict) else str(result)
        detections = result.get("detections", []) if isinstance(result, dict) else []

        # Enrich detections with FASTag & CarInfo API info and optionally save to DB
        enriched: list[dict[str, Any]] = []
        for det in detections:
            plate = det.get("plate")
            carinfo_res = None
            vehicle_details = None
            if plate:
                carinfo_res = carinfo_service.get_vehicle_information(
                    plate,
                    ocr_confidence=float(det.get("confidence") or 1.0),
                    plate_confidence=float(det.get("confidence") or 1.0),
                )
                vehicle_details = database_manager.get_vehicle_info(plate) or carinfo_res
            fastag = database_manager.get_fastag_account(plate) if plate else None
            det["fastag"] = fastag
            det["carinfo"] = carinfo_res
            det["verification_status"] = carinfo_res.get("status") if carinfo_res else "DETECTED"
            det["vehicle_details"] = vehicle_details
            det["owner_name"] = (
                vehicle_details.get("owner_information")
                or vehicle_details.get("owner_name")
                or "Unavailable / Not Verified"
                if vehicle_details
                else "Unavailable / Not Verified"
            )
            det["puc_status"] = (
                vehicle_details.get("pucc_status")
                or vehicle_details.get("puc_status")
                or "N/A"
                if vehicle_details
                else "N/A"
            )
            # compute if balance is sufficient for a sample toll
            vehicle_type = det.get("vehicle_type") or (carinfo_res.get("vehicle_class") if carinfo_res else "Unknown")

            toll_amount = compute_toll_amount(vehicle_type)
            det["toll_amount"] = toll_amount
            det["fastag_sufficient"] = bool(fastag and (float(fastag.get("balance") or 0.0) >= toll_amount))

            # Save detection record if allowed
            if plate and database_manager.can_save_detection(plate, Config.MAX_DUPLICATE_SECONDS):
                database_manager.add_detection(
                    plate_number=plate,
                    vehicle_type=vehicle_type,
                    confidence=float(det.get("confidence") or 0.0),
                    camera_name="Upload Video",
                    vehicle_image=det.get("vehicle_image") or "",
                    plate_image=det.get("plate_image") or "",
                    video_name=filename,
                )
            # Add URLs for template linking
            if det.get("vehicle_image"):
                det["vehicle_image_url"] = url_for("serve_vehicle_image", filename=os.path.basename(det.get("vehicle_image")))
            else:
                det["vehicle_image_url"] = None
            if det.get("plate_image"):
                det["plate_image_url"] = url_for("serve_plate_image", filename=os.path.basename(det.get("plate_image")))
            else:
                det["plate_image_url"] = None
            enriched.append(det)

        # Expose a web URL for the output video if available
        processed_url = None
        if output_path:
            processed_url = url_for("serve_processed", filename=os.path.basename(output_path))

        flash(f"Video processed successfully. Output: {Path(output_path).name}", "success")
        return render_template("upload.html", video_output=processed_url or output_path, video_detections=enriched)
    return render_template("upload.html")


@app.route("/search")
def search() -> Any:
    """Search detections by plate, date, or vehicle type."""
    query = request.args.get("q", "")
    plate = request.args.get("plate", "")
    date = request.args.get("date", "")
    vehicle_type = request.args.get("vehicle_type", "")
    results = database_manager.search_detections(plate=plate or None, date=date or None, vehicle_type=vehicle_type or None)
    if query:
        results = [item for item in results if query.lower() in str(item["plate_number"]).lower()]
    vehicle_record = None
    fastag_account = None
    if plate:
        vehicle_record = database_manager.get_vehicle_record(plate)
        if not vehicle_record:
            # If session allows, enqueue an async RTO lookup
            if session.get("auto_rto"):
                job_id = enqueue_lookup(plate)
                session["last_rto_job"] = job_id
            else:
                # attempt a synchronous fetch into staging
                database_manager.fetch_and_save_rto_record(plate)
                vehicle_record = database_manager.get_vehicle_record(plate)
        fastag_account = database_manager.get_fastag_account(plate)
    return render_template("search.html", results=results, query=query, vehicle_record=vehicle_record, fastag_account=fastag_account)


@app.route('/api/rto-toggle', methods=['POST'])
def api_rto_toggle() -> Any:
    enabled = request.form.get('enabled') == '1'
    session['auto_rto'] = enabled
    return jsonify({'enabled': enabled})


@app.route('/api/rto-enqueue', methods=['POST'])
def api_rto_enqueue() -> Any:
    plate = request.form.get('plate')
    if not plate:
        return jsonify({'error': 'plate required'}), 400
    job_id = enqueue_lookup(plate)
    session['last_rto_job'] = job_id
    return jsonify({'job_id': job_id})


@app.route('/api/rto-job/<job_id>')
def api_rto_job(job_id: str) -> Any:
    status = get_job_status(job_id)
    if status is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(status)


@app.route("/reports")
def reports() -> Any:
    """Render the reports dashboard and support export actions."""
    period = request.args.get("period", "daily")
    rows = database_manager.get_reports(period=period)
    return render_template("reports.html", rows=rows, period=period)


@app.route("/reports/export")
def export_reports() -> Any:
    """Export the current report as CSV, XLSX, or PDF."""
    fmt = request.args.get("format", "csv")
    period = request.args.get("period", "daily")
    rows = database_manager.get_reports(period=period)
    output_path = report_generator.export(rows, fmt=fmt, period=period)
    return send_file(output_path, as_attachment=True, download_name=Path(output_path).name)


@app.route("/blacklist", methods=["GET", "POST"])
def blacklist() -> Any:
    """Manage blacklist entries for suspicious vehicles."""
    if request.method == "POST":
        plate = request.form.get("plate_number", "").strip()
        reason = request.form.get("reason", "")
        if plate:
            database_manager.add_blacklist_entry(plate, reason)
            alert_manager.trigger_alert(plate, reason)
            flash("Blacklist entry added", "success")
        else:
            flash("Plate number is required", "danger")
    entries = database_manager.get_blacklist_entries()
    return render_template("blacklist.html", entries=entries)


@app.route("/blacklist/delete/<int:blacklist_id>")
def delete_blacklist(blacklist_id: int) -> Any:
    """Delete a blacklist entry."""
    database_manager.delete_blacklist_entry(blacklist_id)
    flash("Blacklist entry removed", "info")
    return redirect(url_for("blacklist"))


@app.route("/settings")
def settings() -> Any:
    """Show the configuration settings page."""
    return render_template("settings.html")


@app.route("/toll-plaza", methods=["GET", "POST"])
def toll_plaza() -> Any:
    """Simulate a toll plaza flow using the local registration database."""
    if request.method == "POST":
        plate_number = request.form.get("plate_number", "").strip()
        if not plate_number:
            flash("Plate number is required", "danger")
        else:
            vehicle = database_manager.get_vehicle_record(plate_number)
            if not vehicle:
                flash("Vehicle record not found in local registry", "danger")
            else:
                toll_amount = compute_toll_amount(vehicle.get("vehicle_type", "Unknown"))
                transaction = database_manager.charge_toll(plate_number, toll_amount)
                if transaction is None:
                    flash("Unable to process toll for this vehicle", "danger")
                else:
                    flash(f"Toll transaction created for {plate_number}", "success")
                    return render_template(
                        "toll.html",
                        vehicle=transaction["vehicle"],
                        transaction_id=transaction["transaction_id"],
                        toll_amount=transaction["toll_amount"],
                        balance_before=transaction["fastag_balance_before"],
                        balance_after=transaction["balance_after"],
                        status=transaction["status"],
                        transactions=database_manager.get_toll_transactions(),
                    )
    return render_template("toll.html", transactions=database_manager.get_toll_transactions())


@app.route("/live-camera", methods=["GET", "POST"])
def live_camera() -> Any:
    """Render a page for live camera detection and automatic toll charging."""
    if request.method == "POST" and request.form.get("action") == "add-camera":
        camera_name = request.form.get("camera_name", "").strip()
        camera_type = request.form.get("camera_type", "ip").strip() or "ip"
        camera_url = request.form.get("camera_url", "").strip()
        if camera_name and camera_url:
            camera_manager.add_source(camera_name, camera_type, camera_url)
            flash("Camera source added", "success")
        else:
            flash("Camera name and URL are required", "danger")
    sources = camera_manager.list_sources()
    return render_template("live_camera.html", sources=sources)


@app.route('/processed/<path:filename>')
def serve_processed(filename: str) -> Any:
    return send_from_directory(Config.PROCESSED_FOLDER, filename)


@app.route('/vehicle_images/<path:filename>')
def serve_vehicle_image(filename: str) -> Any:
    return send_from_directory(Config.VEHICLE_IMAGES_FOLDER, filename)


@app.route('/plate_images/<path:filename>')
def serve_plate_image(filename: str) -> Any:
    return send_from_directory(Config.PLATE_IMAGES_FOLDER, filename)


@app.route('/uploads/<path:filename>')
def serve_upload(filename: str) -> Any:
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route('/admin/rto-imports')
def admin_rto_imports() -> Any:
    imports = database_manager.list_rto_imports(only_pending=True)
    return render_template('admin_rto_imports.html', imports=imports)


@app.route('/admin/rto-imports/view/<int:import_id>')
def view_rto_import(import_id: int) -> Any:
    rec = database_manager.get_rto_import(import_id)
    if not rec:
        flash('Import not found', 'danger')
        return redirect(url_for('admin_rto_imports'))
    import json
    payload = json.loads(rec['payload'])
    return render_template('admin_rto_import_view.html', rec=rec, payload=payload)


@app.route('/admin/rto-imports/approve/<int:import_id>')
def approve_rto_import(import_id: int) -> Any:
    ok = database_manager.approve_rto_import(import_id)
    if ok:
        flash('Import approved and saved to vehicle records', 'success')
    else:
        flash('Unable to approve import', 'danger')
    return redirect(url_for('admin_rto_imports'))


from modules.camera_stream import CameraStreamManager

stream_manager = CameraStreamManager()


@app.route("/api/live-detect", methods=["POST"])
def api_live_detect() -> Any:
    """Process a camera frame request and return detection details with decoupled capture and async DB logging."""
    frame = None
    frame_data = request.form.get("frame_data")
    url = request.form.get("url", "0")
    note = request.form.get("note", "").strip()
    try:
        start_time = time.perf_counter()
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fps = 0.0
        if frame_data:
            header, encoded = frame_data.split(",", 1)
            decoded = base64.b64decode(encoded)
            np_img = np.frombuffer(decoded, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Invalid captured image data"}), 400
        else:
            camera_name = next((source["name"] for source in camera_manager.list_sources() if source.get("url") == url), "Selected Camera")
            stream = stream_manager.get_stream(url, name=camera_name)
            fps = stream.fps
            frame, frame_ts = stream.get_latest_frame(timeout=1.0)
            if frame is None or frame.size == 0:
                return jsonify({"error": "Camera frame unavailable or reconnecting"}), 400

        camera_name = next((source["name"] for source in camera_manager.list_sources() if source.get("url") == url), "Selected Camera")
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result = live_processor.process_frame(frame, fps=fps, camera_name=camera_name, detection_time=detection_time, latency_ms=latency_ms)
        processing_fps = 1.0 / max(0.001, time.perf_counter() - start_time)
        fps = fps or processing_fps

        tracking_info = result.get("tracking_info") or {}
        plate_data = result.get("plate") or {}
        plate_text = str(tracking_info.get("plate_number") or plate_data.get("text", ""))
        plate_confidence = float(tracking_info.get("confidence") or plate_data.get("confidence") or 0.0)
        is_finalized = bool(tracking_info.get("is_finalized"))
        is_newly_finalized = bool(tracking_info.get("is_newly_finalized"))

        vehicle_info = None
        transaction = None
        gate_open = False
        blacklist = None
        save_detection = False
        valid_plate = is_valid_plate_text(plate_text)

        carinfo_res = None
        verification_status = "DETECTED"

        if plate_text and (is_finalized or plate_confidence >= Config.OCR_CONFIDENCE_THRESHOLD or valid_plate):
            blacklist = database_manager.get_blacklist_entry(plate_text)
            carinfo_res = carinfo_service.get_vehicle_information(
                plate_text,
                ocr_confidence=plate_confidence,
                plate_confidence=plate_confidence,
            )
            verification_status = carinfo_res.get("status", "VERIFIED" if is_valid_plate_text(plate_text) else "DETECTED")
            vehicle_info = database_manager.get_vehicle_info(plate_text) or carinfo_res

            if vehicle_info:
                v_type = vehicle_info.get("vehicle_class") or vehicle_info.get("vehicle_type") or "Unknown"
                toll_amount = compute_toll_amount(v_type)
                transaction = database_manager.charge_toll(plate_text, toll_amount)
                gate_open = transaction is not None and transaction.get("status") == "Approved"
            status = "Blacklisted" if blacklist else ("Approved" if gate_open else "Denied")
            
            # Save detection ONLY ONCE per vehicle track or window
            if (is_newly_finalized or not tracking_info) and database_manager.can_save_detection(plate_text, Config.MAX_DUPLICATE_SECONDS):
                database_manager.add_detection(
                    plate_number=plate_text,
                    vehicle_type=vehicle_info.get("vehicle_class") or vehicle_info.get("vehicle_type") or result.get("vehicle", {}).get("label", "Unknown"),
                    confidence=plate_confidence,
                    camera_name=camera_name,
                    vehicle_image=result.get("vehicle_image") or "",
                    plate_image=result.get("plate_image") or "",
                    video_name="live_camera" if frame_data else url,
                    lookup_status=verification_status,
                    lookup_error=carinfo_res.get("error") if carinfo_res else "",
                )
                save_detection = True
        else:
            status = "No plate detected"
            verification_status = "LOW CONFIDENCE" if plate_text else "NO DETECT"

        database_manager.add_live_scan_event(
            plate_number=plate_text,
            vehicle_type=vehicle_info.get("vehicle_class") or vehicle_info.get("vehicle_type") if vehicle_info else (result.get("vehicle") or {}).get("label", "Unknown"),
            confidence=plate_confidence,
            camera_name=camera_name,
            source_url=url,
            gate_open=gate_open,
            status=status,
            note=note,
        )

        annotated_image = ''
        if result.get('annotated_frame') is not None:
            annotated_image = _encode_frame_data_uri(result['annotated_frame'])

        return jsonify({
            "vehicle": result.get("vehicle"),
            "plate": result.get("plate"),
            "vehicle_record": vehicle_info,
            "carinfo": carinfo_res,
            "verification_status": verification_status,
            "toll_transaction": transaction,
            "gate_open": gate_open,
            "blacklist": blacklist,
            "status": status,
            "saved": save_detection,
            "annotated_image": annotated_image,

            "detection_time": detection_time,
            "fps": fps,
            "camera_name": camera_name,
            "note": note,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/live-scan-history")
def api_live_scan_history() -> Any:
    """Return recent live-scan events for the history panel."""
    events = database_manager.get_live_scan_events(limit=10)
    return jsonify(events)


@app.route("/api/carinfo/<plate_number>")
@app.route("/api/vehicle/<plate_number>")
def api_carinfo(plate_number: str) -> Any:
    """Return CarInfo-style vehicle and owner details for the given plate."""
    res = carinfo_service.get_vehicle_information(plate_number)
    return jsonify(res)


@app.route("/api/vehicle/lookup", methods=["POST"])
def api_vehicle_lookup() -> Any:
    """Lookup vehicle information using CarInfo service."""
    data = request.get_json(silent=True) or request.form
    plate = (data.get("plate_number") or data.get("plate") or "").strip()
    if not plate:
        return jsonify({"success": False, "error": "Registration plate_number is required"}), 400
    res = carinfo_service.get_vehicle_information(plate)
    return jsonify({
        "success": res.get("verified", False) or res.get("status") == "VERIFIED",
        "plate_number": res.get("registration_number", plate),
        "verified": res.get("verified", False),
        "status": res.get("status"),
        "vehicle": {
            "vehicle_class": res.get("vehicle_class"),
            "manufacturer": res.get("manufacturer"),
            "model": res.get("model"),
            "fuel_type": res.get("fuel_type"),
            "registration_date": res.get("registration_date"),
            "insurance_status": res.get("insurance_status"),
            "insurance_expiry": res.get("insurance_expiry"),
            "fitness_status": res.get("fitness_status"),
            "pucc_status": res.get("pucc_status"),
            "owner_information": res.get("owner_information"),
        },
        "raw_response": res,
    })


@app.route("/api/detections/<int:detection_id>")
def api_detection_detail(detection_id: int) -> Any:
    """Return detail for a specific detection event."""
    with database_manager._connect() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    if not row:
        return jsonify({"error": "Detection not found"}), 404
    det = dict(row)
    plate = det.get("plate_number")
    if plate:
        det["vehicle_info"] = carinfo_service.get_vehicle_information(plate)
    return jsonify(det)


@app.route("/api/health")
def api_health() -> Any:
    """Basic API health endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/spec")
def api_spec() -> Any:
    """Return a minimal Swagger-compatible OpenAPI document."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "ANPR API", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/detections": {"get": {"summary": "List detections"}},
            "/api/vehicle/lookup": {"post": {"summary": "Lookup vehicle details via CarInfo API"}},
        },
    }
    return jsonify(spec)


@app.route("/api/detections")
def api_detections() -> Any:
    """Return recent detections as JSON."""
    rows = database_manager.get_recent_detections(limit=20)
    return jsonify(rows)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)

