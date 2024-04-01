import cv2
import os
from tqdm import tqdm


# Function to extract one frame per second from a video and save them as images
def extract_frames_per_second(video_path, output_folder):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame count
    count = 0

    # Frame index to save
    frame_to_save = 0

    # Loop through frames using tqdm for the progress bar
    for _ in tqdm(range(total_frames), desc="Extracting frames"):
        # Read a frame
        ret, frame = cap.read()

        # If read was successful and it's the frame to save, save the frame as an image
        if ret and count == frame_to_save:
            # Define the output path for the frame image
            frame_path = os.path.join(output_folder, f"frame_{count // fps}.jpg")
            # Save the frame
            cv2.imwrite(frame_path, frame)
            # Update the next frame to save
            frame_to_save += fps
        elif not ret:
            break

        # Increment frame count
        count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {int(count // fps)} frames and saved them to {output_folder}")

extract_frames_per_second('../Videos/13_x11july1615hours.mp4', 'output_frames_folder')
