"""
tracking.py — Module theo dõi quỹ đạo đối tượng
Sử dụng thuật toán DeepSORT / SORT đơn giản
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class Track:
    """Theo dõi một đối tượng qua nhiều frame"""
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0
    hits: int = 1
    miss_count: int = 0
    created_at: float = field(default_factory=time.time)

    def update(self, bbox: Tuple, confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.history.append((cx, cy, time.time()))
        self.hits += 1
        self.miss_count = 0
        self.age += 1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        if len(self.history) < 2:
            return None
        prev = self.history[-2]
        curr = self.history[-1]
        dt = curr[2] - prev[2]
        if dt <= 0:
            return None
        return ((curr[0]-prev[0])/dt, (curr[1]-prev[1])/dt)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def is_accident_risk(self) -> bool:
        """Phát hiện rủi ro dựa trên tốc độ đột ngột thay đổi"""
        if len(self.history) < 5:
            return False
        # Simplified: check if track has abrupt direction change
        vels = []
        hist = list(self.history)
        for i in range(1, len(hist)):
            dt = hist[i][2] - hist[i-1][2]
            if dt > 0:
                vels.append(((hist[i][0]-hist[i-1][0])/dt,
                              (hist[i][1]-hist[i-1][1])/dt))
        if len(vels) < 3:
            return False
        # Sudden deceleration heuristic
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in vels]
        if max(speeds) > 0:
            decel = (max(speeds) - min(speeds[-2:])) / max(speeds)
            return decel > 0.7
        return False


class SimpleTracker:
    """
    SORT-based tracker đơn giản.
    Thực tế: thay bằng deep_sort_realtime hoặc ByteTrack.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: List[dict]) -> List[Track]:
        """
        detections: list of {'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}
        Returns: list of confirmed active tracks
        """
        self.frame_count += 1

        # Increment miss count for all tracks
        for t in self._tracks.values():
            t.miss_count += 1

        matched_det_ids = set()
        matched_track_ids = set()

        # Greedy IoU matching
        for det_idx, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_track_id = None

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = self._iou(det['bbox'], track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                self._tracks[best_track_id].update(det['bbox'], det['confidence'])
                matched_det_ids.add(det_idx)
                matched_track_ids.add(best_track_id)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_ids:
                track = Track(
                    track_id=self._next_id,
                    label=det['label'],
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                )
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                track.history.append((cx, cy, time.time()))
                self._tracks[self._next_id] = track
                self._next_id += 1

        # Remove dead tracks
        dead = [tid for tid, t in self._tracks.items()
                if t.miss_count > self.max_age]
        for tid in dead:
            del self._tracks[tid]

        # Return confirmed tracks only
        return [t for t in self._tracks.values() if t.is_confirmed]

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.is_confirmed]

    @property
    def accident_risk_tracks(self) -> List[Track]:
        return [t for t in self.active_tracks if t.is_accident_risk]

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    @staticmethod
    def _iou(b1: Tuple, b2: Tuple) -> float:
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0
