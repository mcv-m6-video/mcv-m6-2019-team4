from evaluation import (
    mean_ap,
    intersection_over_union,
    iou_vs_time,
)


def run_week1_tasks():
    # Task 1.1: IoU
    intersection_over_union.run()

    # Task 1.2: mAP
    mean_ap.run()

    # Task 2.1: IoU over time
    iou_vs_time.run()


if __name__ == '__main__':
    run_week1_tasks()
