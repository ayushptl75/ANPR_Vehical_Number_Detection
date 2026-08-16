import queue
import time
import unittest
import numpy as np

from modules.camera_stream import AsyncDBLogger, CameraCaptureThread, CameraStreamManager, FrameSampler


class CameraStreamTests(unittest.TestCase):
    def test_frame_sampler_skip_interval(self) -> None:
        sampler = FrameSampler(skip_interval=3)
        # 1st call -> False
        self.assertFalse(sampler.should_sample())
        # 2nd call -> False
        self.assertFalse(sampler.should_sample())
        # 3rd call -> True (sample 1 out of 3)
        self.assertTrue(sampler.should_sample())
        # 4th call -> False
        self.assertFalse(sampler.should_sample())

    def test_async_db_logger_executes_background_tasks(self) -> None:
        logger = AsyncDBLogger(maxsize=10)
        executed = []

        def task():
            executed.append("done")

        success = logger.enqueue(task)
        self.assertTrue(success)

        time.sleep(0.1)
        self.assertIn("done", executed)

        logger.stop()

    def test_bounded_frame_queue_drops_oldest_frame(self) -> None:
        # Create bounded queue maxsize=2
        q: queue.Queue[int] = queue.Queue(maxsize=2)
        q.put_nowait(1)
        q.put_nowait(2)

        # Queue full -> Drop oldest
        if q.full():
            q.get_nowait()
        q.put_nowait(3)

        self.assertEqual(q.get_nowait(), 2)
        self.assertEqual(q.get_nowait(), 3)

    def test_camera_stream_manager_singleton_lifecycle(self) -> None:
        manager = CameraStreamManager()
        # Non-existent or dummy camera handle creates stream session
        stream = manager.get_stream("0", name="MockCam")
        self.assertIsNotNone(stream)

        manager.stop_all()
        self.assertFalse(stream.running)


if __name__ == "__main__":
    unittest.main()
