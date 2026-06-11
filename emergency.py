# -*- coding: utf-8 -*-
"""
Emergency incident registry and dispatch planning.

This module keeps an in-memory incident list for the dashboard demo. It also
builds a simulated response plan with the nearest traffic police and ambulance
unit using Haversine distance.
"""

import math
import time
import json
from pathlib import Path


INCIDENTS = []
VALID_STATUSES = {"NEW", "VERIFIED", "DISPATCHED", "RESOLVED", "FALSE_ALARM"}
FEEDBACK_DIR = Path(__file__).resolve().parent / "feedback_cases"


CSGT_UNITS = [
    {"id": "CSGT_Q1", "name": "Doi CSGT Quan 1", "lat": 10.7769, "lng": 106.7009, "hotline": "028-3829-3829", "district": "Quan 1"},
    {"id": "CSGT_Q3", "name": "Doi CSGT Quan 3", "lat": 10.7763, "lng": 106.6903, "hotline": "028-3932-3232", "district": "Quan 3"},
    {"id": "CSGT_BTH", "name": "Doi CSGT Binh Thanh", "lat": 10.8120, "lng": 106.7050, "hotline": "028-3512-2020", "district": "Binh Thanh"},
    {"id": "CSGT_TB", "name": "Doi CSGT Tan Binh", "lat": 10.8020, "lng": 106.6520, "hotline": "028-3849-0101", "district": "Tan Binh"},
    {"id": "CSGT_TD", "name": "Doi CSGT TP Thu Duc", "lat": 10.7700, "lng": 106.7300, "hotline": "028-3740-5050", "district": "Thu Duc"},
    {"id": "CSGT_Q10", "name": "Doi CSGT Quan 10", "lat": 10.7740, "lng": 106.6680, "hotline": "028-3865-2323", "district": "Quan 10"},
    {"id": "CSGT_PC08", "name": "PC08 - CSGT TP.HCM", "lat": 10.7970, "lng": 106.6860, "hotline": "19006969", "district": "TP.HCM"},
]


AMBULANCE_UNITS = [
    {"id": "AMB_CR", "name": "BV Cho Ray - Cap cuu", "lat": 10.7552, "lng": 106.6584, "hotline": "028-3855-4137", "district": "Quan 5"},
    {"id": "AMB_NDG", "name": "BV Nhan Dan Gia Dinh", "lat": 10.8042, "lng": 106.6978, "hotline": "028-3841-0411", "district": "Binh Thanh"},
    {"id": "AMB_BD", "name": "BV Binh Dan", "lat": 10.7799, "lng": 106.6919, "hotline": "028-3822-7723", "district": "Quan 3"},
    {"id": "AMB_TN", "name": "BV Thong Nhat", "lat": 10.7984, "lng": 106.6786, "hotline": "028-3864-0898", "district": "Tan Binh"},
    {"id": "AMB_LVT", "name": "BV Le Van Thinh - Thu Duc", "lat": 10.7812, "lng": 106.7388, "hotline": "028-3740-2320", "district": "Thu Duc"},
    {"id": "AMB_115", "name": "Trung tam Cap cuu 115", "lat": 10.7872, "lng": 106.6968, "hotline": "115", "district": "TP.HCM"},
    {"id": "AMB_ND115", "name": "BV Nhan Dan 115", "lat": 10.7740, "lng": 106.6840, "hotline": "028-3865-4249", "district": "Quan 10"},
]


def _haversine(lat1, lng1, lat2, lng2) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _find_nearest(units: list[dict], lat: float, lng: float) -> dict:
    best = None
    best_dist = float("inf")
    for unit in units:
        dist = _haversine(lat, lng, unit["lat"], unit["lng"])
        if dist < best_dist:
            best = unit
            best_dist = dist

    if best is None:
        return {}

    # Demo ETA: urban average speed ~25 km/h plus 2 minutes preparation time.
    eta_min = round(best_dist / 25 * 60 + 2, 1)
    return {
        **best,
        "distance_km": round(best_dist, 2),
        "eta_minutes": eta_min,
        "eta_str": f"~{int(round(eta_min))} phut",
    }


def build_dispatch_plan(camera_lat: float, camera_lng: float) -> dict:
    return {
        "csgt": _find_nearest(CSGT_UNITS, camera_lat, camera_lng),
        "ambulance": _find_nearest(AMBULANCE_UNITS, camera_lat, camera_lng),
    }


def _incident_priority(ensemble_result: dict) -> int:
    if "priority_score" in ensemble_result:
        try:
            return int(ensemble_result["priority_score"])
        except (TypeError, ValueError):
            pass
    score = float(ensemble_result.get("ensemble_score") or 0) * 70
    votes = min(int(ensemble_result.get("votes") or 0), 3)
    score += votes / 3 * 30
    return int(max(0, min(100, round(score))))


def _feedback_label(status: str) -> str | None:
    if status == "VERIFIED":
        return "confirmed_accident"
    if status == "FALSE_ALARM":
        return "false_alarm"
    return None


def _save_feedback_case(incident: dict, status: str) -> None:
    label = _feedback_label(status)
    if not label:
        return
    FEEDBACK_DIR.mkdir(exist_ok=True)
    record = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "incident_id": incident.get("id"),
        "camera_id": incident.get("camera_id"),
        "camera_name": incident.get("camera_name"),
        "ensemble_score": incident.get("ensemble_score"),
        "priority_score": incident.get("priority_score"),
        "votes": incident.get("votes"),
        "reason": incident.get("reason"),
        "models": incident.get("models"),
        "source": incident.get("source"),
        "video_url": incident.get("video_url"),
        "has_image": bool(incident.get("image")),
    }
    target = FEEDBACK_DIR / f"{label}.jsonl"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_incident(camera: dict, ensemble_result: dict, image_b64=None, source=None) -> dict:
    priority = _incident_priority(ensemble_result)
    for incident in INCIDENTS:
        if incident.get("camera_id") == camera["id"] and incident.get("status") not in {"RESOLVED", "FALSE_ALARM"}:
            incident.update({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": "ACCIDENT",
                "ensemble_score": ensemble_result["ensemble_score"],
                "priority_score": priority,
                "votes": ensemble_result["votes"],
                "reason": ensemble_result["reason"],
                "models": ensemble_result["models"],
                "cascade": ensemble_result.get("cascade"),
                "video_url": ensemble_result.get("video_url") or incident.get("video_url"),
                "image": image_b64 or incident.get("image"),
                "source": source or incident.get("source"),
                "dispatch_plan": incident.get("dispatch_plan") or build_dispatch_plan(camera["lat"], camera["lng"]),
            })
            INCIDENTS.remove(incident)
            INCIDENTS.insert(0, incident)
            return incident

    incident = {
        "id": f"INC-{int(time.time() * 1000)}",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": camera["id"],
        "camera_name": camera["name"],
        "lat": camera["lat"],
        "lng": camera["lng"],
        "status": "NEW",
        "level": "ACCIDENT",
        "ensemble_score": ensemble_result["ensemble_score"],
        "priority_score": priority,
        "votes": ensemble_result["votes"],
        "reason": ensemble_result["reason"],
        "models": ensemble_result["models"],
        "cascade": ensemble_result.get("cascade"),
        "video_url": ensemble_result.get("video_url"),
        "image": image_b64,
        "source": source,
        "dispatch_plan": build_dispatch_plan(camera["lat"], camera["lng"]),
        "dispatch_log": [],
    }
    INCIDENTS.insert(0, incident)
    return incident


def get_incidents() -> list:
    return sorted(
        INCIDENTS,
        key=lambda inc: (
            inc.get("status") in {"RESOLVED", "FALSE_ALARM"},
            -int(inc.get("priority_score") or 0),
            inc.get("time", ""),
        ),
    )


def get_feedback_stats() -> dict:
    FEEDBACK_DIR.mkdir(exist_ok=True)
    out = {}
    for label in ("confirmed_accident", "false_alarm"):
        path = FEEDBACK_DIR / f"{label}.jsonl"
        if not path.exists():
            out[label] = 0
            continue
        with path.open("r", encoding="utf-8") as fh:
            out[label] = sum(1 for line in fh if line.strip())
    out["dir"] = str(FEEDBACK_DIR)
    return out


def update_status(incident_id: str, status: str) -> dict | None:
    status = str(status or "").upper()
    if status not in VALID_STATUSES:
        return None

    for incident in INCIDENTS:
        if incident["id"] == incident_id:
            incident["status"] = status
            incident["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_feedback_case(incident, status)
            return incident
    return None


def dispatch_incident(incident_id: str) -> dict | None:
    for incident in INCIDENTS:
        if incident["id"] != incident_id:
            continue

        plan = incident.get("dispatch_plan", {})
        log_entry = {
            "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "csgt": plan.get("csgt", {}).get("name", "N/A"),
            "ambulance": plan.get("ambulance", {}).get("name", "N/A"),
            "operator": "Dashboard",
        }
        incident.setdefault("dispatch_log", []).append(log_entry)
        incident["status"] = "DISPATCHED"
        incident["dispatched_at"] = log_entry["dispatched_at"]
        return incident
    return None


def get_emergency_units() -> dict:
    return {"csgt": CSGT_UNITS, "ambulance": AMBULANCE_UNITS}
