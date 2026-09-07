import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.core.metrics import MetricsCollector


def track(track_id=7, name='alice'):
    return SimpleNamespace(tracker_id=track_id, name=name, state=1, identity_state=2,
                           missed_frames=0, occlusion_frames=0)


class MetricsCollectorTests(unittest.TestCase):
    def test_json_csv_and_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = MetricsCollector(output_dir=temp, flush_size=1)
            collector.start_session('V08.mp4', 30., {'identity_lock_timeout': 180})
            person = track()
            collector.record('DETECTION', 1)
            collector.record('TRACK_CREATED', 1, track=person)
            collector.record('IDENTITY_CONFIRMED', 2, track=person)
            collector.record('RECOVERY_ATTEMPT', 3, track=person)
            collector.record('RECOVERY_SUCCESS', 3, track=person)
            collector.record('GALLERY_UPDATE_ATTEMPT', 3, track=person)
            collector.record('GALLERY_UPDATE_ACCEPTED', 3, track=person)
            collector.record('PRESENCE_IN', 3, identity='alice')
            collector.finalize(3, None, [person], 30.)
            data = json.loads((collector.run_dir / 'metrics.json').read_text())
            self.assertEqual(data['metrics']['total_detections'], 1)
            self.assertEqual(data['metrics']['unique_tracker_ids'], 1)
            self.assertEqual(data['metrics']['successful_recoveries'], 1)
            self.assertIsNone(data['metrics']['false_in_events'])
            self.assertTrue((collector.run_dir / 'events.csv').exists())

    def test_sessions_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            first = MetricsCollector(output_dir=temp); first.start_session('V09.mp4', 30., {}); first.finalize(0, None, [], 30.)
            second = MetricsCollector(output_dir=temp); second.start_session('V09.mp4', 30., {}); second.finalize(0, None, [], 30.)
            self.assertNotEqual(first.run_dir, second.run_dir)

    def test_event_logging_off_keeps_aggregate_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = MetricsCollector(output_dir=temp, event_logging=False)
            collector.start_session('V08.mp4', 30., {})
            collector.record('DETECTION', 1)
            collector.finalize(1, None, [], 30.)
            data = json.loads((collector.run_dir / 'metrics.json').read_text())
            self.assertEqual(data['metrics']['total_detections'], 1)
            self.assertFalse((collector.run_dir / 'events.csv').exists())

    def test_disabled_creates_no_files(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = MetricsCollector(enabled=False, output_dir=temp)
            collector.start_session('V08.mp4', 30., {}); collector.finalize(0, None, [], 30.)
            self.assertEqual(list(Path(temp).rglob('*')), [])


if __name__ == '__main__': unittest.main()
