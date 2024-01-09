

import time
import sys

def three_step_process_with_progress_bar():
    total_steps = 3

    def update_progress_bar(step):
        progress = (step / total_steps) * 100
        print(f"\rProgress: [{'#' * step + ' ' * (total_steps - step)}] {progress:.2f}%", end='')
        sys.stdout.flush()

    # Step 1
    print("Starting Step 1...")
    time.sleep(3)  # Wait for 3 seconds
    print("\nStep 1 Completed.")
    update_progress_bar(1)

    # Step 2
    print("Starting Step 2...")
    time.sleep(3)  # Wait for 3 seconds
    print("\nStep 2 Completed.")
    update_progress_bar(2)

    # Step 3
    print("Starting Step 3...")
    time.sleep(3)  # Wait for 3 seconds
    print("\nStep 3 Completed.")
    update_progress_bar(3)

    print("\nAll steps completed.")





three_step_process_with_progress_bar()