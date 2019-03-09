from evaluation import (
    mean_ap,
)

from week1 import task1, task2, task3, task4


def run_week1_tasks():
    """
    - Task 1: Detection metrics.
    - Task 2: Detection metrics. Temporal analysis.
    - Task 3: Optical flow evaluation metrics.
    - Task 4: Visual representation optical flow.
    """
    # Task 1.1: IoU
    task1.test_iou_with_synth_data()
    task1.test_iou_with_noise()

    # Task 1.2: mAP
    mean_ap.compute_average_precision()

    # Task 2.1: IoU over time
    task2.iou_vs_time()

    # Task 3 (all subtasks together)
    task3.test_optical_flow_metrics()

    # Task 4
    task4.visualise_optical_flow()


if __name__ == '__main__':
    run_week1_tasks()
