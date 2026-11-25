"""Behavioral regression tests; no camera or trained model required."""
import unittest
import numpy as np
from fitness_core import ExerciseTracker, LandmarkEMA, FEATURE_COLUMNS, extract_features, visible


def features(**overrides):
    data = {"knee": 175., "elbow": 175., "back": 165., "body": 175., "torso_horizontal": 1.,
            "shoulder_left": 15., "shoulder_right": 15., "ankle_ratio": 0.8, "wrists_above_head": 0.}
    from fitness_core import METRIC_COLUMNS
    for key in METRIC_COLUMNS:
        data.setdefault(key, 175.)
    data.update(overrides)
    return data


class MotionTests(unittest.TestCase):
    def hold(self, tracker, f, start, label=None, confidence=1.):
        for offset in (0., 0.1, 0.2):
            tracker.update(f, start + offset, label, confidence)

    def test_squat_counts_only_complete_cycle(self):
        t = ExerciseTracker("squat")
        self.hold(t, features(knee=90), 0)
        self.hold(t, features(), .3)
        self.assertEqual(t.reps, 0)
        self.hold(t, features(knee=90), .6)
        self.hold(t, features(), .9)
        self.assertEqual((t.reps, t.correct_reps), (1, 1))
        self.hold(t, features(), 1.2)
        self.assertEqual(t.reps, 1)

    def test_squat_checks_deepest_point_not_transient_good_frame(self):
        t = ExerciseTracker("squat")
        self.hold(t, features(), 0)
        self.hold(t, features(knee=90), .3)
        self.hold(t, features(knee=50), .6)
        self.hold(t, features(), .9)
        self.assertEqual((t.reps, t.correct_reps), (1, 0))

    def test_pushup_body_and_elbow(self):
        for body, expected in [(175, 1), (140, 0)]:
            t = ExerciseTracker("pushup")
            self.hold(t, features(), 0)
            self.hold(t, features(elbow=85, body=body), .3)
            self.hold(t, features(), .6)
            self.assertEqual((t.reps, t.correct_reps), (1, expected))

    def test_missing_pose_cannot_bridge_a_rep(self):
        t = ExerciseTracker("squat")
        self.hold(t, features(), 0)
        self.hold(t, features(knee=90), .3)
        t.invalidate(.6)
        self.hold(t, features(), .7)
        self.assertEqual(t.reps, 0)

    def test_low_confidence_or_other_exercise_blocks_reps(self):
        for label, confidence in [("squat_up", .3), ("jj_closed", .99)]:
            t = ExerciseTracker("squat")
            self.hold(t, features(), 0, label, confidence)
            self.hold(t, features(knee=90), .3, label, confidence)
            self.hold(t, features(), .6, label, confidence)
            self.assertEqual(t.reps, 0)

    def test_ai_wrong_prevents_correct_rep(self):
        t = ExerciseTracker("squat")
        self.hold(t, features(), 0, "squat_up")
        self.hold(t, features(knee=90), .3, "squat_down_wrong")
        self.hold(t, features(), .6, "squat_up")
        self.assertEqual((t.reps, t.correct_reps), (1, 0))

    def test_jumping_jacks_requires_arms_wrists_and_feet(self):
        for wrists, ratio, expected in [(1, 1.8, 1), (0, 1.8, 0), (1, .9, 0)]:
            t = ExerciseTracker("jj")
            self.hold(t, features(), 0)
            self.hold(t, features(shoulder_left=160, shoulder_right=160, wrists_above_head=wrists, ankle_ratio=ratio), .3)
            self.hold(t, features(), .6)
            self.assertEqual((t.reps, t.correct_reps), (1, expected))

    def test_plank_never_counts_bad_pose_standing_or_missing_time(self):
        t = ExerciseTracker("plank")
        self.hold(t, features(), 0)
        self.assertAlmostEqual(t.seconds, .2)
        t.update(features(body=140), .3)
        t.update(features(torso_horizontal=0), .4)
        t.invalidate(.5)
        t.update(features(), 10)
        self.assertAlmostEqual(t.seconds, .2)
        t.update(features(), 10.1)
        self.assertAlmostEqual(t.seconds, .3)

    def test_fast_threshold_jitter_does_not_count(self):
        t = ExerciseTracker("squat")
        self.hold(t, features(), 0)
        for i in range(20):
            t.update(features(knee=90 if i % 2 else 175), .3 + i * .02)
        self.assertEqual(t.reps, 0)


class FeatureTests(unittest.TestCase):
    def pose(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(.25, .75, (33, 4))
        p[:, 3] = .99
        p[11, :2], p[12, :2] = [.35, .25], [.65, .25]
        p[23, :2], p[24, :2] = [.4, .55], [.6, .55]
        return p

    def test_schema_and_aspect_ratio(self):
        p = self.pose()
        f = extract_features(p, 640, 480)
        self.assertEqual(set(f), set(FEATURE_COLUMNS))
        # Same pixels on a wider canvas must preserve every feature.
        q = p.copy()
        q[:, 0] /= 2
        q[:, 2] /= 2
        g = extract_features(q, 1280, 480)
        np.testing.assert_allclose([f[k] for k in FEATURE_COLUMNS], [g[k] for k in FEATURE_COLUMNS])

    def test_visibility_does_not_linger_after_smoothing(self):
        p = self.pose()
        ema = LandmarkEMA()
        ema.update(p)
        p[15, 3] = 0
        q = ema.update(p)
        self.assertFalse(visible(q, "jj"))
        self.assertEqual(q[15, 3], 0)


if __name__ == "__main__":
    unittest.main()
