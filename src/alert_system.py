"""
alert_system.py — Module gửi thông báo (Email/API)
Hệ thống cảnh báo khẩn cấp thời gian thực
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    EMAIL = "Email"
    SMS   = "SMS"
    API   = "REST API"
    PUSH  = "Push Notification"


@dataclass
class Alert:
    alert_id: str
    level: AlertLevel
    camera_id: str
    location: str
    event_type: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    channels_sent: List[str] = field(default_factory=list)
    acknowledged: bool = False
    response_time_ms: Optional[float] = None

    @property
    def datetime_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S %d/%m/%Y")

    def to_dict(self) -> dict:
        d = asdict(self)
        d['level'] = self.level.value
        d['timestamp'] = self.datetime_str
        return d


class AlertSystem:
    """
    Hệ thống cảnh báo khẩn cấp đa kênh.
    Hỗ trợ: Email, SMS, REST API, Push Notification.
    """

    def __init__(self,
                 min_confidence: float = 0.7,
                 cooldown_seconds: float = 30.0,
                 channels: Optional[List[AlertChannel]] = None):
        self.min_confidence = min_confidence
        self.cooldown = cooldown_seconds
        self.channels = channels or [AlertChannel.API, AlertChannel.EMAIL]
        self._alert_history: List[Alert] = []
        self._last_alert_time: dict = {}  # camera_id -> timestamp
        self._callbacks: List[Callable] = []
        self._alert_counter = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable[[Alert], None]):
        """Đăng ký callback khi có cảnh báo mới"""
        self._callbacks.append(fn)

    def process_detection(self,
                          camera_id: str,
                          location: str,
                          event_type: str,
                          confidence: float,
                          algorithm: str = "YOLOv8") -> Optional[Alert]:
        """
        Xử lý kết quả nhận diện và gửi cảnh báo nếu đủ điều kiện.
        Returns: Alert object nếu cảnh báo được gửi, None nếu không.
        """
        if confidence < self.min_confidence:
            return None

        if not self._passes_cooldown(camera_id):
            logger.debug(f"Skipping alert for {camera_id} — cooldown active")
            return None

        level = self._determine_level(event_type, confidence)
        alert = self._create_alert(camera_id, location, event_type, confidence, level)

        self._dispatch(alert)
        self._alert_history.append(alert)
        self._last_alert_time[camera_id] = time.time()

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.response_time_ms = (time.time() - alert.timestamp) * 1000
                return True
        return False

    @property
    def unacknowledged_alerts(self) -> List[Alert]:
        return [a for a in self._alert_history if not a.acknowledged]

    @property
    def recent_alerts(self) -> List[Alert]:
        cutoff = time.time() - 3600  # last hour
        return [a for a in self._alert_history if a.timestamp >= cutoff]

    def get_stats(self) -> dict:
        total = len(self._alert_history)
        acked = sum(1 for a in self._alert_history if a.acknowledged)
        critical = sum(1 for a in self._alert_history
                       if a.level == AlertLevel.CRITICAL)
        avg_rt = None
        rts = [a.response_time_ms for a in self._alert_history
               if a.response_time_ms is not None]
        if rts:
            avg_rt = sum(rts) / len(rts)

        return {
            "total_alerts": total,
            "acknowledged": acked,
            "pending": total - acked,
            "critical": critical,
            "avg_response_ms": avg_rt,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _passes_cooldown(self, camera_id: str) -> bool:
        last = self._last_alert_time.get(camera_id, 0)
        return (time.time() - last) >= self.cooldown

    def _determine_level(self, event_type: str, confidence: float) -> AlertLevel:
        if event_type.lower() in ("accident", "crash", "collision") and confidence >= 0.85:
            return AlertLevel.CRITICAL
        elif confidence >= 0.70:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    def _create_alert(self, camera_id, location, event_type,
                      confidence, level) -> Alert:
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:05d}-{int(time.time())}"
        return Alert(
            alert_id=alert_id,
            level=level,
            camera_id=camera_id,
            location=location,
            event_type=event_type,
            confidence=confidence,
        )

    def _dispatch(self, alert: Alert):
        """Gửi cảnh báo qua tất cả kênh đã đăng ký"""
        for channel in self.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                elif channel == AlertChannel.SMS:
                    self._send_sms(alert)
                elif channel == AlertChannel.API:
                    self._call_api(alert)
                elif channel == AlertChannel.PUSH:
                    self._send_push(alert)
                alert.channels_sent.append(channel.value)
            except Exception as e:
                logger.error(f"Failed to send via {channel.value}: {e}")

        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _send_email(self, alert: Alert):
        """Giả lập gửi email — thay bằng smtplib hoặc SendGrid thực tế"""
        logger.info(f"[EMAIL] {alert.level.value}: {alert.event_type} at {alert.location} "
                    f"| Conf: {alert.confidence:.1%} | {alert.datetime_str}")

    def _send_sms(self, alert: Alert):
        """Giả lập SMS — thay bằng Twilio thực tế"""
        logger.info(f"[SMS] EMERGENCY: Accident detected at {alert.location} "
                    f"({alert.confidence:.0%})")

    def _call_api(self, alert: Alert):
        """Giả lập REST API call — thay bằng requests.post thực tế"""
        payload = json.dumps(alert.to_dict(), ensure_ascii=False, indent=2)
        logger.info(f"[API] POST /emergency-alerts\n{payload}")

    def _send_push(self, alert: Alert):
        """Giả lập push notification"""
        logger.info(f"[PUSH] 🚨 {alert.event_type} detected | {alert.location}")
