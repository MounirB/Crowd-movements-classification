import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random


# This function is used for now by confusion_matrix.py
def generate_video_sample(video_clip_path, input_shape):
    """
    Returns a video clip depending on its type
    :param video_clip_path: unique path to the RGB video clip, double path to the two parts of a flow clip,
    or triple path for the rgb clip and the two parts of the flow clip
    :param input_shape: the input shape of the video clip that should determine if it's flow, rgb or joint
    :return: RGB or Flow video clip
    """

    # Check channels to determine the video clip type
    if input_shape[3] == 2: # Flow clip
        x_axis_flow_clip = get_flow_videoclips(os.path.join(video_clip_path[0]),
                                               input_shape[0], input_shape[1], input_shape[2])
        y_axis_flow_clip = get_flow_videoclips(os.path.join(video_clip_path[1]),
                                               input_shape[0], input_shape[1], input_shape[2])
        # We expand the dimensions of x_axis_flow_clip and y_axis_flow_clip
        # We do this to add each channel of the flow clip next to the other
        x_axis_flow_clip = np.expand_dims(x_axis_flow_clip, axis=3)
        y_axis_flow_clip = np.expand_dims(y_axis_flow_clip, axis=3)
        clip = np.append(x_axis_flow_clip, y_axis_flow_clip, axis=3)

        # Expand clip dimensions for the target architecture
        clip = np.expand_dims(clip, axis=0)
    elif input_shape[3] == 3: #RGB clip
        clip = get_rgb_videoclip(os.path.join(video_clip_path), input_shape[0], input_shape[1], input_shape[2])

        # Expand clip dimensions for the target architecture
        clip = np.expand_dims(clip, axis=0)
    else: #Joint clip
        # Get flow clip
        x_axis_flow_clip = get_flow_videoclips(os.path.join(video_clip_path[1]),
                                               input_shape[0], input_shape[1], input_shape[2])
        y_axis_flow_clip = get_flow_videoclips(os.path.join(video_clip_path[2]),
                                               input_shape[0], input_shape[1], input_shape[2])
        # We expand the dimensions of x_axis_flow_clip and y_axis_flow_clip
        # We do this to add each channel of the flow clip next to the other
        x_axis_flow_clip = np.expand_dims(x_axis_flow_clip, axis=3)
        y_axis_flow_clip = np.expand_dims(y_axis_flow_clip, axis=3)
        flow_clip = np.append(x_axis_flow_clip, y_axis_flow_clip, axis=3)

        # Get RGB clip
        rgb_clip = get_rgb_videoclip(os.path.join(video_clip_path[0]),
                                      input_shape[0], input_shape[1], input_shape[2])

        # Get clip
        rgb_clip = np.expand_dims(rgb_clip, axis=0)
        flow_clip = np.expand_dims(flow_clip, axis=0)

        clip = [rgb_clip, flow_clip]

    return clip


def generate_video_clips(video_data, model_type, input_shape, num_classes, batch_size):
    """
    Generates video clips to feed fit_generator
    :param data: video clips or joint video clips
    :param model_type: mentions the type of the deep learning architecture
    :param input_shape: input shape of this form (Frames_number, Frame_height, Frame_width, Channels)
    or joint input shapes [rgb_inputshape, flow_inputshape]
    :param num_classes: number of classes
    :param batch_size: the number of video clips to train
    :return: yields a video clip or a joint video clips generator for fit_generator
    """
    if model_type == 'TWOSTREAM_I3D':
        while True:
            # Randomize the position of the indices to make an array
            videoclips_indices_array = np.random.permutation(video_data.count()[0])
            for batch in range(0, len(videoclips_indices_array), batch_size):
                # Create a current batch container to stack clip(s) and frame(s)
                current_batch = videoclips_indices_array[batch:(batch + batch_size)]

                # Initializing the stacks (batches) x_train (clips, frames) and y_train (labels)
                # y_train : labels
                labels = np.empty([0], dtype=np.int32)


                # x_train
                flow_channels = 2
                rgb_channels = 3
                flow_clips = np.empty([0, input_shape[0], input_shape[1], input_shape[2], flow_channels],
                                     dtype=np.float32)
                rgb_clips = np.empty([0, input_shape[0], input_shape[1], input_shape[2], rgb_channels],
                                    dtype=np.float32)



                for index in current_batch:
                    single_rgb_clip = get_rgb_videoclip(os.path.join(video_data['rgbclips_path'].values[index].strip()),
                        input_shape[0], input_shape[1], input_shape[2])

                    x_axis_flow_clip = get_flow_videoclips(os.path.join(video_data['x_axis_flowclips_path'].values[index].strip()),
                        input_shape[0], input_shape[1], input_shape[2])
                    y_axis_flow_clip = get_flow_videoclips(os.path.join(video_data['y_axis_flowclips_path'].values[index].strip()),
                        input_shape[0], input_shape[1], input_shape[2])
                    # We expand the dimensions of x_axis_flow_clip and y_axis_flow_clip so that we can add each channel of the flow clip next to the other
                    x_axis_flow_clip = np.expand_dims(x_axis_flow_clip, axis=3)
                    y_axis_flow_clip = np.expand_dims(y_axis_flow_clip, axis=3)
                    single_flow_clip = np.append(x_axis_flow_clip, y_axis_flow_clip, axis=3)

                    single_label = video_data['class'].values[index]

                    # We expand the dimensions of single_clip so that we can stack the batch of clips
                    single_rgb_clip = np.expand_dims(single_rgb_clip, axis=0)
                    single_flow_clip = np.expand_dims(single_flow_clip, axis=0)

                    # Appending them to existing batch
                    flow_clips = np.append(flow_clips, single_flow_clip, axis=0)
                    rgb_clips = np.append(rgb_clips, single_rgb_clip, axis=0)

                    # We append labels to the existing batch of labels
                    labels = np.append(labels, [single_label])
                # Convert the label expression to binary with to_categorical
                labels = to_categorical(labels, num_classes=num_classes)

                yield ([rgb_clips, flow_clips], labels)
    else: # all other architecture types
        while True:
            # Randomize the indices to make an array
            videoclips_indices_array = np.random.permutation(video_data.count()[0])
            for batch in range(0, len(videoclips_indices_array), batch_size):
                # Create a current batch container to stack clip(s) and frame(s)
                current_batch = videoclips_indices_array[batch:(batch + batch_size)]

                # Initializing the stacks (batches) x_train (clips) and y_train (labels)
                # y_train : labels
                labels = np.empty([0], dtype=np.int32)

                # x_train
                rgb_channels = 3
                clips = np.empty([0, input_shape[0], input_shape[1], input_shape[2], rgb_channels], dtype=np.float32)

                for index in current_batch:
                    # get video clip and label
                    single_clip = get_rgb_videoclip(os.path.join(video_data['rgbclips_path'].values[index].strip()),
                                                                 input_shape[0], input_shape[1], input_shape[2])
                    single_label = video_data['class'].values[index]

                    # We expand the dimensions of single_clip so that we can stack each one of them on the batch
                    single_clip = np.expand_dims(single_clip, axis=0)

                    # We append clips to the existing batch of clips
                    clips = np.append(clips, single_clip, axis=0)

                    # We append labels to the existing batch of labels
                    labels = np.append(labels, [single_label])
                # Convert the label expression to binary with to_categorical
                labels = to_categorical(labels, num_classes=num_classes)

                yield ([clips], labels)

def select_frames(frames, frames_per_video):
    """
    Select a certain number of frames determined by the number (frames_per_video)
    :param frames: list of frames
    :param frames_per_video: number of frames to select
    :return: selection of frames
    """
    step = len(frames)//frames_per_video
    if step == 0:
        step = 1
    first_frames_selection = frames[::step]
    final_frames_selection = first_frames_selection[:frames_per_video]

    return final_frames_selection


def get_rgb_videoclip(rgb_videoclip, frames_per_video, frame_height, frame_width):
    """
    From an RGB channeled video clip returns a random frame and a number of frames indicated by frames_per_video
    :param rgb_videoclip: the source video clip in RGB
    :param frames_per_video: number of frames per video to select
    :param frame_height: frame height
    :param frame_width: frame width
    :return: selected number of frames
    """
    cap = cv2.VideoCapture(rgb_videoclip)

    frames = list()
    if not cap.isOpened():
        cap.open(rgb_videoclip)
    ret = True
    while (True and ret):
        ret, frame = cap.read()

        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # The following operations are intended to select a precise number of frames
    # and to resize them according to the decided setup of frame_height/width
    selected_frames = select_frames(frames, frames_per_video)

    # Resizing frames to fit the decided setup
    resized_selected_frames = list()
    for selected_frame in selected_frames:
        resized_selected_frame = cv2.resize(selected_frame, (frame_width, frame_height))
        resized_selected_frames.append(resized_selected_frame)

    # return frame, video_clip
    return np.asarray(resized_selected_frames)

def get_flow_videoclips(flow_videoclip, frames_per_video, frame_height, frame_width):
    """
    From an RGB channeled flow video clip returns a random frame and a number of frames indicated by frames_per_video
    :param flow_videoclip: the source grayed video clip, but in RGB
    :param frames_per_video: number of frames per video to select
    :param frame_height: frame height
    :param frame_width: frame width
    :return: selected number of one-channeled frames
    """
    capture = cv2.VideoCapture(flow_videoclip)

    # Extract flow frames
    flow_frames = list()
    if not capture.isOpened():
        capture.open(flow_videoclip)
    ret = True
    while (True and ret):
        ret, three_channeled_flow_frame = capture.read()
        if ret:
            flow_frame = cv2.cvtColor(three_channeled_flow_frame, cv2.COLOR_BGR2GRAY)
            flow_frames.append(flow_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()

    """
    The following operations are intended to select a precise number of frames, and 
    to resize them according to the decided setup of frame_height/width
    """
    selected_flow_frames = select_frames(flow_frames, frames_per_video)
    # Resizing frames to fit the decided setup
    resized_selected_flow_frames = list()
    for selected_flow_frame in selected_flow_frames:
        resized_selected_flow_frame = cv2.resize(selected_flow_frame, (frame_width, frame_height))
        resized_selected_flow_frames.append(resized_selected_flow_frame)

    # return flow_videoclip
    return np.asarray(resized_selected_flow_frames)

