"""High-level orchestration of detection, tracking, presence accounting, and storage."""
from __future__ import annotations
import cv2
import time
from pathlib import Path
from app.config import Settings, database_timestamp
from app.core.detector import PersonDetector
from app.core.reid import FeatureExtractor
from app.core.face import FaceProcessor
from app.core.gallery import Gallery
from app.core.tracker import IdentityState, ReIDTracker, TrackState
from app.core.presence import PresenceTracker
from app.core.metrics import MetricsCollector
from app.storage.mysql_store import MySQLPresenceStore
from app.utils.utils import draw_label, draw_presence_panel
from app.utils.pipeline_logging import get_pipeline_logger

logger = get_pipeline_logger()


class ReIDPipeline:
    """Build the ML pipeline once and apply it to each frame of a video source."""
    def __init__(self, settings: Settings):
        """Create the detector, embedding models, gallery, tracker, and optional MySQL store."""
        self.settings = settings
        self.extractor = FeatureExtractor(
            model_name=settings.reid_model,
            model_path=settings.reid_weights,
            device=settings.device,
        )
        self.face_processor = FaceProcessor(
            model_name=settings.face_model,
            det_size=settings.face_det_size,
        )
        self.gallery = Gallery.load(
            settings.gallery_dir,
            body_extractor=self.extractor,
            face_processor=self.face_processor
        )
        self.detector = PersonDetector(
            model_name=str(settings.detector_model),
            confidence=settings.confidence,
            device=settings.device,
        )
        self.tracker = ReIDTracker(
            gallery=self.gallery,
            extractor=self.extractor,
            face_processor=self.face_processor,
            threshold=settings.match_threshold,
            w_body=settings.hybrid_weight_body,
            w_face=settings.hybrid_weight_face,
            max_lost_frames=settings.tracker_max_lost_frames,
            max_occlusion_frames=settings.tracker_max_occlusion_frames,
            max_recovery_frames=settings.tracker_max_recovery_frames,
            max_trajectory_history=settings.tracker_max_trajectory_history,
            max_body_gallery=settings.tracker_max_body_gallery,
            max_face_gallery=settings.tracker_max_face_gallery,
            occlusion_iou_threshold=settings.tracker_occlusion_iou_threshold,
            recovery_reid_threshold=settings.tracker_recovery_reid_threshold,
            motion_gate_threshold=settings.tracker_motion_gate_threshold,
            recovery_motion_gate=settings.tracker_recovery_motion_gate,
            turn_threshold=settings.tracker_turn_threshold,
            motion_confidence_threshold=settings.tracker_motion_confidence_threshold,
            gallery_update_threshold=settings.tracker_gallery_update_threshold,
            normal_reid_gate=settings.tracker_normal_reid_gate,
            detection_dedup_iou=settings.tracker_detection_dedup_iou,
            min_detection_confidence=settings.tracker_min_detection_confidence,
            min_detection_width=settings.tracker_min_detection_width,
            min_detection_height=settings.tracker_min_detection_height,
            min_detection_area=settings.tracker_min_detection_area,
            duplicate_track_iou=settings.tracker_duplicate_track_iou,
            duplicate_appearance_threshold=settings.tracker_duplicate_appearance_threshold,
            identity_face_threshold=settings.tracker_identity_face_threshold,
            identity_body_candidate_threshold=settings.tracker_identity_body_candidate_threshold,
            identity_body_confirm_threshold=settings.tracker_identity_body_confirm_threshold,
            identity_margin=settings.tracker_identity_margin,
            identity_candidate_min_frames=settings.tracker_identity_candidate_min_frames,
            identity_retention_frames=settings.tracker_identity_retention_frames,
            identity_lock_timeout=settings.tracker_identity_lock_timeout,
            recovery_margin=settings.tracker_recovery_margin,
            association_min_iou=settings.tracker_association_min_iou,
            association_max_scale_change=settings.tracker_association_max_scale_change,
            gallery_min_detection_confidence=settings.tracker_gallery_min_detection_confidence,
        )
        self.presence = PresenceTracker()
        self.metrics = MetricsCollector(settings.metrics_enabled, settings.metrics_event_logging,
                                        settings.metrics_output_dir, settings.metrics_log_detections,
                                        settings.metrics_flush_size)
        self.tracker.metrics = self.metrics
        self._metrics_presence_events = 0
        self.store = None
        if settings.mysql_host:
            self.store = MySQLPresenceStore(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
            )
            logger.info('Pipeline initialized: device=%s, database=%s', settings.device, bool(self.store))

    def run_on_video(self, source: str | int, save_to_db: bool = True,
                     output_video: Path | None = None, max_frames: int = -1,
                     preview: bool = True) -> None:
        """Process frames until the source ends or the user presses Q in the preview window."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error('Unable to open source: %s', source)
            raise RuntimeError(f'Cannot open source: {source}')
        logger.info('Pipeline started: source=%s, save_to_db=%s, output=%s', source, save_to_db, output_video)
        started_at = database_timestamp()
        prev_time = database_timestamp()
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.metrics.start_session(str(source), fps, self._metrics_configuration())
        out = None
        processing_failed = False
        try:
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video), fourcc, fps,
                                      (int(cap.get(3)), int(cap.get(4))))
                if not out.isOpened():
                    logger.error('Unable to create output video: %s', output_video)
                    raise RuntimeError(f'Cannot create output video: {output_video}')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if max_frames > 0 and frame_count > max_frames:
                    break
                current_time = database_timestamp()
                # Use real elapsed time rather than frame count so durations work at any FPS.
                elapsed = (current_time - prev_time).total_seconds()
                prev_time = current_time
                detections = self.detector.detect(frame)
                for detection in detections:
                    self.metrics.record('DETECTION', frame_count, frame_count / fps,
                                        detection_confidence=float(getattr(detection, 'confidence', 0.)))
                tracks = self.tracker.update(frame, detections)
                # Only confirmed gallery matches contribute to attendance statistics.
                # Predicted/stale tracks are not currently visible; counting them here
                # would keep a departed person's timer running until track expiry.
                visible_names = {
                    t.name for t in tracks
                    if t.name != 'Unknown' and (
                        (t.time_since_update == 0 and t.state in (TrackState.CONFIRMED, TrackState.RECOVERED))
                        or (t.state == TrackState.OCCLUDED and t.missed_frames <= self.settings.tracker_max_occlusion_frames)
                    )
                }
                self.presence.update(visible_names, elapsed, current_time)
                # Presence remains unchanged; observe only newly emitted events.
                while getattr(self, '_metrics_presence_events', 0) < len(self.presence.events):
                    event = self.presence.events[self._metrics_presence_events]; self._metrics_presence_events += 1
                    self.metrics.record(f'PRESENCE_{event.event_type}', frame_count, frame_count / fps, identity=event.person_name, presence_event=event.event_type)
                state_names = {
                    TrackState.TENTATIVE: 'TENTATIVE', TrackState.CONFIRMED: 'CONFIRMED',
                    TrackState.OCCLUDED: 'OCCLUDED', TrackState.RECOVERED: 'RECOVERED',
                    TrackState.LOST: 'LOST', TrackState.DELETED: 'DELETED',
                }
                identity_names = {
                    IdentityState.UNKNOWN: 'UNKNOWN', IdentityState.CANDIDATE: 'CANDIDATE',
                    IdentityState.CONFIRMED: 'IDENTITY', IdentityState.RETAINED: 'RETAINED',
                }
                for track in tracks:
                    # A predicted box is useful visual feedback during an occlusion, but
                    # stale normal/lost tracks should not be drawn as currently visible.
                    if track.time_since_update != 0 and (track.state != TrackState.OCCLUDED or track.missed_frames > self.settings.tracker_max_visual_occlusion_frames):
                        continue
                    known = track.state in (TrackState.CONFIRMED, TrackState.RECOVERED) and track.name != 'Unknown'
                    label = track.name if known else f'Person {track.tracker_id}'
                    state = state_names[track.state]
                    detail = f'ID {track.tracker_id} | {state} | conf {track.association_confidence:.2f}'
                    if track.identity_state != IdentityState.UNKNOWN:
                        detail += f' | {identity_names[track.identity_state]}:{track.identity_source or track.identity_candidate}'
                    if track.state == TrackState.OCCLUDED and track.occluded_by is not None:
                        detail += f' | by {track.occluded_by}'
                    draw_label(frame, tuple(track.bbox.astype(int)), label, known, detail)
                draw_presence_panel(frame, self.presence.records, self.presence.active_names)
                if out is not None:
                    out.write(frame)

                if preview:
                    cv2.imshow("ReID Tracking", frame)
                if preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info('User requested pipeline stop.')
                    break
        except Exception:
            processing_failed = True
            logger.exception('Pipeline failed: source=%s, frames=%s', source, frame_count)
            raise
        finally:
            # Always release hardware/files, even when detection raises an exception.
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            completed_at = database_timestamp()
            self.presence.finalize(completed_at)
            while getattr(self, '_metrics_presence_events', 0) < len(self.presence.events):
                event = self.presence.events[self._metrics_presence_events]; self._metrics_presence_events += 1
                self.metrics.record(f'PRESENCE_{event.event_type}', frame_count, frame_count / fps, identity=event.person_name, presence_event=event.event_type)
            self.metrics.finalize(frame_count, self.presence, self.tracker.tracks, fps)
            if self.metrics.enabled:
                summary = self.metrics.summary
                print('\nV06 TRACKING METRICS\n'
                      f'Video: {source}\nFrames: {frame_count}\nFPS: {fps:.2f}\n'
                      f"Total detections: {summary['total_detections']}\nUnique tracker IDs: {summary['unique_tracker_ids']}\n"
                      f"Confirmed identities: {summary['confirmed_identities']}\nTrack fragments: {summary['track_fragments']}\n"
                      f"ID switches: {summary['id_switches']}\nFalse identity claims: {summary['false_identity_claims']}\n"
                      f"Recovery attempts: {summary['recovery_attempts']}\nSuccessful recoveries: {summary['successful_recoveries']}\n"
                      f"Failed recoveries: {summary['failed_recoveries']}\nAmbiguous recoveries: {summary['ambiguous_recoveries']}\n"
                      f"IN events: {summary['in_events']}\nOUT events: {summary['out_events']}\n"
                      f"Gallery updates accepted/rejected: {summary['gallery_updates_accepted']}/{summary['gallery_updates_rejected']}\n"
                      f"Association rejections: {summary['association_rejections']}\nProcessing FPS: {summary['processing_fps']:.2f}\n"
                      f'Metrics JSON: {self.metrics.run_dir / "metrics.json"}\nEvents CSV: {self.metrics.csv_path}')
            if self.store and save_to_db:
                try:
                    self.store.save_run(
                        source=str(source),
                        started_at=started_at,
                        completed_at=completed_at,
                        records=self.presence.records,
                        events=self.presence.events,
                    )
                    logger.info('Run saved: source=%s, frames=%s, people=%s, events=%s',
                                source, frame_count, len(self.presence.records), len(self.presence.events))
                except Exception as error:
                    logger.exception('Database save failed for source=%s', source)
                    if not processing_failed:
                        raise RuntimeError('Pipeline finished, but the run could not be saved to MySQL.') from error
            if self.store:
                try:
                    self.store.close()
                except Exception:
                    logger.exception('Failed to close the database connection.')
            logger.info('Pipeline finished: source=%s, frames=%s, failed=%s', source, frame_count, processing_failed)

    def _metrics_configuration(self) -> dict[str, object]:
        """Non-secret settings needed to reproduce a metrics run."""
        keys = ('tracker_motion_gate_threshold', 'tracker_association_min_iou', 'tracker_association_max_scale_change',
                'tracker_recovery_reid_threshold', 'tracker_recovery_margin', 'tracker_identity_lock_timeout',
                'tracker_gallery_update_threshold', 'tracker_identity_candidate_min_frames', 'tracker_max_occlusion_frames')
        return {key: getattr(self.settings, key) for key in keys}
