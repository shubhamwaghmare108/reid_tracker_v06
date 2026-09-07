"""Runtime configuration for the person re-identification pipeline."""
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')


def _setting(name: str, default: str) -> str:
    """Read a setting from environment variables, then Streamlit secrets, then a default."""
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
        mysql_secrets = st.secrets.get('mysql', {})
        if name in mysql_secrets:
            return str(mysql_secrets[name])
    except Exception:
        pass
    return default



def database_timestamp() -> datetime:
    """Return a naive UTC value suitable for MySQL DATETIME storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class Settings:
    """One place for model paths, matching thresholds, and optional database credentials."""
    project_root: Path = Path(_setting('REID_PROJECT_ROOT', str(PROJECT_ROOT)))
    detector_model: str = Path('Models') / 'yolo11s-seg.pt'
    reid_model: str = 'osnet_x1_0'
    reid_weights: str = Path('Models') / 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    device: str = 'cpu'
    confidence: float = 0.1
    match_threshold: float = 0.55
    known_retention_threshold: float = float(_setting('REID_KNOWN_RETENTION_THRESHOLD', '0.45'))
    gallery_dir: Path | None = None
    output_dir: Path | None = None

    hybrid_weight_body: float = 0.5
    hybrid_weight_face: float = 0.5

    # Tracker values deliberately separate short-term motion matching from strict
    # long-term identity recovery.  They can be overridden by callers/settings
    # without changing detector or extractor interfaces.
    tracker_max_lost_frames: int = int(_setting('REID_TRACKER_MAX_LOST_FRAMES', '30'))
    tracker_max_occlusion_frames: int = int(_setting('REID_TRACKER_MAX_OCCLUSION_FRAMES', '45'))
    tracker_max_recovery_frames: int = int(_setting('REID_TRACKER_MAX_RECOVERY_FRAMES', '180'))
    tracker_max_trajectory_history: int = int(_setting('REID_TRACKER_MAX_TRAJECTORY_HISTORY', '30'))
    tracker_max_body_gallery: int = int(_setting('REID_TRACKER_MAX_BODY_GALLERY', '12'))
    tracker_max_face_gallery: int = int(_setting('REID_TRACKER_MAX_FACE_GALLERY', '6'))
    tracker_occlusion_iou_threshold: float = float(_setting('REID_TRACKER_OCCLUSION_IOU_THRESHOLD', '0.15'))
    tracker_recovery_reid_threshold: float = float(_setting('REID_TRACKER_RECOVERY_REID_THRESHOLD', '0.78'))
    tracker_motion_gate_threshold: float = float(_setting('REID_TRACKER_MOTION_GATE_THRESHOLD', '4.0'))
    tracker_turn_threshold: float = float(_setting('REID_TRACKER_TURN_THRESHOLD', '0.75'))
    tracker_motion_confidence_threshold: float = float(_setting('REID_TRACKER_MOTION_CONFIDENCE_THRESHOLD', '0.45'))
    tracker_gallery_update_threshold: float = float(_setting('REID_TRACKER_GALLERY_UPDATE_THRESHOLD', '0.72'))
    tracker_normal_reid_gate: float = float(_setting('REID_TRACKER_NORMAL_REID_GATE', '0.20'))
    tracker_recovery_motion_gate: float = float(_setting('REID_TRACKER_RECOVERY_MOTION_GATE', '8.0'))
    tracker_detection_dedup_iou: float = float(_setting('REID_TRACKER_DETECTION_DEDUP_IOU', '0.75'))
    tracker_min_detection_confidence: float = float(_setting('REID_TRACKER_MIN_DETECTION_CONFIDENCE', '0.35'))
    tracker_min_detection_width: int = int(_setting('REID_TRACKER_MIN_DETECTION_WIDTH', '20'))
    tracker_min_detection_height: int = int(_setting('REID_TRACKER_MIN_DETECTION_HEIGHT', '40'))
    tracker_min_detection_area: int = int(_setting('REID_TRACKER_MIN_DETECTION_AREA', '1200'))
    tracker_duplicate_track_iou: float = float(_setting('REID_TRACKER_DUPLICATE_TRACK_IOU', '0.70'))
    tracker_duplicate_appearance_threshold: float = float(_setting('REID_TRACKER_DUPLICATE_APPEARANCE_THRESHOLD', '0.80'))
    tracker_max_visual_occlusion_frames: int = int(_setting('REID_TRACKER_MAX_VISUAL_OCCLUSION_FRAMES', '12'))
    tracker_identity_face_threshold: float = float(_setting('REID_TRACKER_IDENTITY_FACE_THRESHOLD', '0.72'))
    tracker_identity_body_candidate_threshold: float = float(_setting('REID_TRACKER_IDENTITY_BODY_CANDIDATE_THRESHOLD', '0.72'))
    tracker_identity_body_confirm_threshold: float = float(_setting('REID_TRACKER_IDENTITY_BODY_CONFIRM_THRESHOLD', '0.84'))
    tracker_identity_margin: float = float(_setting('REID_TRACKER_IDENTITY_MARGIN', '0.08'))
    tracker_identity_candidate_min_frames: int = int(_setting('REID_TRACKER_IDENTITY_CANDIDATE_MIN_FRAMES', '4'))
    tracker_identity_retention_frames: int = int(_setting('REID_TRACKER_IDENTITY_RETENTION_FRAMES', '45'))
    # V06: identity ownership outlives a short-lived tracker.  These settings are
    # deliberately additive: the V05 recognition thresholds above are unchanged.
    tracker_identity_lock_timeout: int = int(_setting('REID_TRACKER_IDENTITY_LOCK_TIMEOUT', '180'))
    tracker_recovery_margin: float = float(_setting('REID_TRACKER_RECOVERY_MARGIN', '0.08'))
    tracker_association_min_iou: float = float(_setting('REID_TRACKER_ASSOCIATION_MIN_IOU', '0.01'))
    tracker_association_max_scale_change: float = float(_setting('REID_TRACKER_ASSOCIATION_MAX_SCALE_CHANGE', '2.85'))
    tracker_gallery_min_detection_confidence: float = float(_setting('REID_TRACKER_GALLERY_MIN_DETECTION_CONFIDENCE', '0.70'))

    metrics_enabled: bool = _setting('METRICS_ENABLED', 'true').lower() == 'true'
    metrics_event_logging: bool = _setting('METRICS_EVENT_LOGGING', 'true').lower() == 'true'
    metrics_output_dir: Path = Path(_setting('METRICS_OUTPUT_DIR', 'results'))
    metrics_log_detections: bool = _setting('METRICS_LOG_DETECTIONS', 'false').lower() == 'true'
    metrics_flush_size: int = int(_setting('METRICS_FLUSH_SIZE', '500'))

    face_model: str = 'buffalo_l'
    face_det_size: tuple[int, int] = (640, 640)
    face_det_threshold: float = 0.5

    mysql_host: str = _setting('REID_MYSQL_HOST', '')
    mysql_port: int = int(_setting('REID_MYSQL_PORT', '3306'))
    mysql_user: str = _setting('REID_MYSQL_USER', '')
    mysql_password: str = _setting('REID_MYSQL_PASSWORD', '')
    mysql_database: str = _setting('REID_MYSQL_DATABASE', '')

    def __post_init__(self) -> None:
        """Fill derived folders and turn relative model paths into absolute project paths."""
        self.gallery_dir = self.gallery_dir or self.project_root / 'known_people'
        self.output_dir = self.output_dir or self.project_root / 'output'
        detector_path = Path(self.detector_model)
        weights_path = Path(self.reid_weights)
        if not detector_path.is_absolute():
            self.detector_model = str(self.project_root / detector_path)
        if not weights_path.is_absolute():
            self.reid_weights = str(self.project_root / weights_path)
        if not self.metrics_output_dir.is_absolute():
            self.metrics_output_dir = self.project_root / self.metrics_output_dir
