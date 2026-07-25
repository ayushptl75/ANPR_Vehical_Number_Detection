from __future__ import annotations

from flask import Blueprint, render_template, request, redirect, url_for, flash
from modules.database_manager import DatabaseManager
from modules.utils import clean_plate_text

vehicle_bp = Blueprint('vehicle_bp', __name__, url_prefix='/admin/vehicles')

db = DatabaseManager()


@vehicle_bp.route('/', methods=['GET'])
def list_vehicles():
    q = request.args.get('q', '').strip()
    with db._connect() as conn:
        if q:
            rows = conn.execute("SELECT * FROM vehicles WHERE plate_number LIKE ? ORDER BY id DESC", (f"%{q}%",)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM vehicles ORDER BY id DESC LIMIT 200").fetchall()
    vehicles = [dict(r) for r in rows]
    return render_template('admin/vehicles.html', vehicles=vehicles, query=q)


@vehicle_bp.route('/add', methods=['GET', 'POST'])
def add_vehicle():
    if request.method == 'POST':
        plate = clean_plate_text(request.form.get('plate_number', ''))
        if not plate:
            flash('Registration number is required', 'danger')
            return redirect(url_for('vehicle_bp.add_vehicle'))
        data = {
            'vehicle_type': request.form.get('vehicle_type'),
            'manufacturer': request.form.get('manufacturer'),
            'model': request.form.get('model'),
            'fuel_type': request.form.get('fuel_type'),
            'registration_date': request.form.get('registration_date'),
            'registration_state': request.form.get('registration_state'),
            'insurance_status': request.form.get('insurance_status'),
            'insurance_expiry': request.form.get('insurance_expiry'),
            'puc_status': request.form.get('puc_status'),
            'puc_expiry': request.form.get('puc_expiry'),
            'owner_name': request.form.get('owner_name'),
            'rc_status': request.form.get('rc_status'),
        }
        db.update_vehicle(plate, data)
        flash('Vehicle record saved', 'success')
        return redirect(url_for('vehicle_bp.list_vehicles'))
    return render_template('admin/vehicle_form.html', vehicle=None)


@vehicle_bp.route('/edit/<plate>', methods=['GET', 'POST'])
def edit_vehicle(plate):
    plate_norm = clean_plate_text(plate)
    if request.method == 'POST':
        data = {
            'vehicle_type': request.form.get('vehicle_type'),
            'manufacturer': request.form.get('manufacturer'),
            'model': request.form.get('model'),
            'fuel_type': request.form.get('fuel_type'),
            'registration_date': request.form.get('registration_date'),
            'registration_state': request.form.get('registration_state'),
            'insurance_status': request.form.get('insurance_status'),
            'insurance_expiry': request.form.get('insurance_expiry'),
            'puc_status': request.form.get('puc_status'),
            'puc_expiry': request.form.get('puc_expiry'),
            'owner_name': request.form.get('owner_name'),
            'rc_status': request.form.get('rc_status'),
            'owner_name': request.form.get('owner_name'),
            'vehicle_color': request.form.get('vehicle_color'),
        }
        db.update_vehicle(plate_norm, data)
        flash('Vehicle updated', 'success')
        return redirect(url_for('vehicle_bp.list_vehicles'))
    rec = db.get_vehicle_info(plate_norm)
    return render_template('admin/vehicle_form.html', vehicle=rec)


@vehicle_bp.route('/delete/<plate>', methods=['POST'])
def delete_vehicle(plate):
    plate_norm = clean_plate_text(plate)
    db.delete_vehicle(plate_norm)
    flash('Vehicle record deleted', 'info')
    return redirect(url_for('vehicle_bp.list_vehicles'))
