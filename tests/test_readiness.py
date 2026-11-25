"""실제 카메라 없이 합성 원본 Image Landmark로 준비 휴리스틱을 검증한다."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from app.core import readiness as r


def t_pose():
    """정면 사용자의 신체 LEFT(11)가 원본 화면 오른쪽인 T-Pose다."""
    points = [dict(x=.5, y=.1, visibility=.9) for _ in range(33)]
    for index, xy in {0:(.5,.12),11:(.6,.3),12:(.4,.3),13:(.72,.3),14:(.28,.3),
                      15:(.85,.3),16:(.15,.3),23:(.56,.55),24:(.44,.55),25:(.56,.7),
                      26:(.44,.7),27:(.56,.85),28:(.44,.85),31:(.58,.9),32:(.42,.9)}.items():
        points[index].update(x=xy[0], y=xy[1])
    return points


class ReadinessTests(unittest.TestCase):
    def evaluate(self, points, width=640, height=480):
        return r.evaluate_readiness(points, width, height)

    def test_valid_t_pose_all_conditions(self):
        result = self.evaluate(t_pose())
        self.assertTrue(all(result[key] for key in ('pose_detected','full_body_visible','arms_horizontal','arms_extended','arms_spread','ready')))
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['left_elbow_angle_deg'], 180)

    def test_no_pose(self):
        result = self.evaluate(None)
        self.assertFalse(result['ready'])
        self.assertFalse(result['pose_detected'])
        self.assertEqual(result['reasons'], ['NO_POSE'])

    def test_wrong_landmark_count(self):
        for points in ([], t_pose()[:32], t_pose()+[{}]):
            self.assertEqual(self.evaluate(points)['reasons'], ['INVALID_LANDMARK_COUNT'])

    def test_nonfinite_major_coordinates(self):
        for axis in ('x','y'):
            for value in (np.nan,np.inf,-np.inf):
                points=t_pose();points[11][axis]=value
                self.assertEqual(self.evaluate(points)['reasons'], ['INVALID_LANDMARKS'])

    def test_low_visibility(self):
        for index in r.REQUIRED_LANDMARKS:
            points=t_pose();points[index]['visibility']=.49
            result=self.evaluate(points)
            self.assertFalse(result['ready'])
            self.assertIn('LOW_VISIBILITY',result['reasons'])

    def test_nonessential_points_need_not_be_visible(self):
        points=t_pose()
        for index in set(range(33))-set(r.REQUIRED_LANDMARKS):
            points[index] = None
        self.assertTrue(self.evaluate(points)['ready'])

    def test_body_outside_and_margin(self):
        for index in (0,11,15,23,25,27):
            for value in (-.1,.02,.98,1.1):
                points=t_pose();points[index]['y']=value
                result=self.evaluate(points)
                self.assertFalse(result['ready'])
                self.assertIn('BODY_OUT_OF_FRAME',result['reasons'])

    def test_feet_outside(self):
        for index in (31,32):
            points=t_pose();points[index]['x']=1.01
            self.assertIn('BODY_OUT_OF_FRAME',self.evaluate(points)['reasons'])

    def test_each_arm_horizontal_independently(self):
        for index in (13,14,15,16):
            points=t_pose();points[index]['y']=.45
            result=self.evaluate(points)
            self.assertFalse(result['arms_horizontal'])
            self.assertFalse(result['ready'])

    def test_bent_elbows(self):
        for index in (13,14):
            points=t_pose();points[index]['y']=.34
            result=self.evaluate(points)
            self.assertIn('ARMS_NOT_EXTENDED',result['reasons'])
            self.assertFalse(result['ready'])

    def test_wrists_inside_body(self):
        for index in (15,16):
            points=t_pose();points[index]['x']=.5
            result=self.evaluate(points)
            self.assertFalse(result['arms_spread'])
            self.assertIn('ARMS_NOT_SPREAD',result['reasons'])

    def test_insufficient_outward_reach(self):
        points=t_pose();points[15]['x']=.73;points[13]['x']=.67
        result=self.evaluate(points)
        self.assertTrue(result['arms_extended'])
        self.assertFalse(result['arms_spread'])

    def test_original_left_right_indices_and_input_unchanged(self):
        points=t_pose();points[13]['y']=.4
        original=deepcopy(points)
        result=self.evaluate(points)
        self.assertLess(result['left_elbow_angle_deg'],155)
        self.assertEqual(result['right_elbow_angle_deg'],180)
        self.assertEqual(points,original)

    def test_aspect_ratio_affects_pixel_geometry(self):
        points=t_pose();points[13]['y']=.34
        wide=self.evaluate(points,1280,480)
        tall=self.evaluate(points,480,1280)
        self.assertTrue(wide['ready'])
        self.assertFalse(tall['ready'])
        self.assertGreater(wide['left_elbow_angle_deg'],tall['left_elbow_angle_deg'])

    def test_resolution_scaling_preserves_geometry(self):
        points=t_pose();points[15]['y']=.31
        self.assertEqual(self.evaluate(points,640,480),self.evaluate(points,1280,960))

    def test_objects_and_three_column_array(self):
        points=t_pose()
        objects=[SimpleNamespace(**point) for point in points]
        array=np.array([[p['x'],p['y'],p['visibility']] for p in points])
        self.assertEqual(self.evaluate(points),self.evaluate(objects))
        self.assertEqual(self.evaluate(points),self.evaluate(array))

    def test_missing_visibility_not_assumed(self):
        points=t_pose();del points[11]['visibility']
        self.assertFalse(self.evaluate(points)['ready'])
        points=[SimpleNamespace(x=.5,y=.5,z=.1) for _ in range(33)]
        self.assertEqual(self.evaluate(points)['reasons'],['INVALID_LANDMARKS'])

    def test_invalid_visibility_and_types(self):
        for value in (None,True,'0.9',np.nan,1.1,-.1):
            points=t_pose();points[0]['visibility']=value
            self.assertEqual(self.evaluate(points)['reasons'],['INVALID_LANDMARKS'])

    def test_bad_dimensions(self):
        for width,height in [(0,480),(640,-1),(np.nan,480),(640,np.inf),(True,480),('640',480)]:
            self.assertEqual(self.evaluate(t_pose(),width,height)['reasons'],['INVALID_IMAGE_SIZE'])

    def test_degenerate_shoulders_and_arm(self):
        points=t_pose();points[11]['x']=points[12]['x']
        self.assertIn('DEGENERATE_GEOMETRY',self.evaluate(points)['reasons'])
        points=t_pose();points[13]=points[11].copy()
        result=self.evaluate(points)
        self.assertFalse(result['ready'])
        self.assertIsNone(result['left_elbow_angle_deg'])

    def test_visibility_and_margin_boundaries_inclusive(self):
        points=t_pose()
        for point in points:point['visibility']=r.MIN_VISIBILITY
        points[0]['y']=r.FRAME_MARGIN
        self.assertTrue(self.evaluate(points)['ready'])


if __name__=='__main__':
    unittest.main()
