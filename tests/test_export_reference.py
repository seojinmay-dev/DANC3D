"""Avatar export: catalog discovery, safe reruns, units and playback timing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preprocessing import export_reference as exporter


class ReferenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw/labels/001_TEST'
        self.source.mkdir(parents=True)
        self.output = self.root / 'public/001_TEST.json'
        self.song = dict(id='001_TEST', title='test', artist='artist',
                         raw_label_dir='raw/labels/001_TEST', raw_source_dir='raw/images/001_TEST')
        (self.root / 'songs.json').write_text(json.dumps([self.song]))
        self.write_frames([10, 12, 14])

    def write_frames(self, indices):
        (self.source / 'frames.json').write_text(json.dumps({
            'videos': [{'frame_rate': 60, 'file_name': '001_TEST.mp4'}],
            'annotations': [dict(frame_index=i, keypoints3d=list(range(87))) for i in indices],
        }))

    def run_batch(self, *args):
        with patch.object(exporter, 'DATA_ROOT', self.root), \
             patch.object(exporter, 'OUTPUT_DIR', self.output.parent), \
             patch('sys.argv', ['export_reference.py', *args]):
            exporter.main()

    def test_default_catalog_export_units_and_timing(self):
        self.run_batch()
        payload = json.loads(self.output.read_text())
        self.assertEqual(payload['fps'], 30)
        self.assertEqual([f['timestamp'] for f in payload['frames']], [0, 1/30, 2/30])
        self.assertEqual(payload['frames'][0]['joints']['leftShoulder'], [.42, .43, .44])
        self.assertEqual(payload['metadata']['title'], 'test')

    def test_valid_output_skips_reading_source_and_preserves_file(self):
        self.run_batch()
        before = (self.output.read_bytes(), self.output.stat().st_mtime_ns)
        with patch.object(exporter, 'load_frames', side_effect=AssertionError('must skip')):
            self.run_batch()
        self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))

    def test_broken_output_regenerated_and_force_reexports_valid_output(self):
        self.output.parent.mkdir()
        self.output.write_text('{truncated')
        self.run_batch()
        self.assertTrue(exporter.valid_existing(self.output, '001_TEST'))
        self.song['title'] = 'changed'
        (self.root / 'songs.json').write_text(json.dumps([self.song]))
        self.run_batch('--sequence', '001_TEST', '--force')
        self.assertEqual(json.loads(self.output.read_text())['metadata']['title'], 'changed')

    def test_irregular_timing_keeps_previous_output(self):
        self.run_batch()
        before = self.output.read_bytes()
        self.write_frames([10, 12, 16])
        with self.assertRaises(SystemExit) as error:
            self.run_batch('--force')
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
