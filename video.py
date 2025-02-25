import os

import ffmpeg


def merge_videos(input_folder: str, output_file: str = "lunar_lander_nul.mp4"):
    """
    Merges all MP4 videos in the specified folder into a single video using ffmpeg.

    :param input_folder: Path to the folder containing the video files.
    :param output_file: Name of the output merged video file.
    """
    # Get a sorted list of all video files
    video_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp4")])

    if not video_files:
        print("No video files found in the folder.")
        return

    input_paths = [ f for f in video_files]

    # Create an input list file for ffmpeg
    list_file = os.path.join(input_folder, "input_list.txt")
    with open(list_file, "w") as f:
        for file in input_paths:
            f.write(f"file '{file}'\n")

    # Run ffmpeg command to concatenate videos
    ffmpeg.input(list_file, format='concat', safe=0).output(output_file, c='copy').run(overwrite_output=True)

    print(f"Merged video saved as {output_file}")
