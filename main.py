from week1 import task1, task2, task3, task4
from week2 import task1, task2, task3, task4

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
    task1.compute_mAP()

    # Task 2.1: IoU over time
    task2.iou_vs_time()

    # Task 3 (all subtasks together)
    task3.test_optical_flow_metrics()

    # Task 4
    task4.visualise_optical_flow()


def run_week2_tasks():
    # Task 1: Single Gaussian modeling.
    task1.bg_segmentation_single_gaussian(video_name="simple_gaussian", alpha=1)
    task1.bg_segmentation_single_gaussian(
        video_name="simple_gaussian_prepostproc",
        preproc=True,
        postproc=True
    )

    # Task 2: Adaptive Single Gaussian modeling
    task2.bg_segmentation_single_gaussian_adaptive(
        video_name="simple_gaussian_adaptive",
        preproc=True,
        postproc=True
    )

    # Task 3: State of the art
    task3.compare_state_of_the_art()

    # Task 4: Color space conversions
    task4.compare_color_spaces()

def run_week3_tasks():
    from week3 import task2_1, task2_2

    task2_1.overlap_tracking()
    task2_2.kalman_tracking()


if __name__ == '__main__':
    #run_week1_tasks()
    #run_week2_tasks()
    run_week3_tasks()