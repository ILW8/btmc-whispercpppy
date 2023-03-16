import random
import subprocess
import sys
import time
from pathlib import Path

import streamlink
import ffmpeg  # python-ffmpeg

from yep import run_once, print_callback

# def start_ffmpeg_process1(in_filename):
#     logger.info('Starting ffmpeg process1')
#     args = (
#         ffmpeg
#         .input(in_filename)
#         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#         .compile()
#     )
#     return subprocess.Popen(args, stdout=subprocess.PIPE)
#
#
# def start_ffmpeg_process2(out_filename, width, height):
#     logger.info('Starting ffmpeg process2')
#     args = (
#         ffmpeg
#         .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
#         .output(out_filename, pix_fmt='yuv420p')
#         .overwrite_output()
#         .compile()
#     )
#     return subprocess.Popen(args, stdin=subprocess.PIPE)

# ...

#     process1 = start_ffmpeg_process1(in_filename)
#     process2 = start_ffmpeg_process2(out_filename, width, height)

# ...

# process1 = (
#     ffmpeg
#     .input(in_filename)
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=8)
#     .run_async(pipe_stdout=True)
# )
#
# process2 = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
#     .output(out_filename, pix_fmt='yuv420p')
#     .overwrite_output()
#     .run_async(pipe_stdin=True)
# )
#
# while True:
#     in_bytes = process1.stdout.read(width * height * 3)
#     if not in_bytes:
#         break
#     in_frame = (
#         np
#         .frombuffer(in_bytes, np.uint8)
#         .reshape([height, width, 3])
#     )
#
#     # See examples/tensorflow_stream.py:
#     out_frame = deep_dream.process_frame(in_frame)
#
#     process2.stdin.write(
#         out_frame
#         .astype(np.uint8)
#         .tobytes()
#     )
#
# process2.stdin.close()
# process1.wait()
# process2.wait()

from tqdm import tqdm
from functools import partial

from ctypes import c_bool
from multiprocessing import Process, Value
import os
import tempfile

CHANNEL_NAME = "jynxzi"
TEMP_DATA_DIR_TD = tempfile.TemporaryDirectory()
TEMP_DATA_DIR = TEMP_DATA_DIR_TD.name


def consume_ephemeral_data(files_directory: str, file_prefix: str, stop: Value):
    while not stop.value:
        files = os.listdir(Path(files_directory))
        if len(files) > 1:  # 2 files ensures the first file is already closed from writing
            print(f"Current files in queue: {len(files)}")
            current_file = os.path.join(files_directory, sorted(files)[0])
            # print("processing " + current_file)
            try:
                run_once(current_file, print_callback)
                os.unlink(current_file)
            except UnicodeDecodeError:
                pass
        sys.stdout.flush()

        time.sleep(0.1)
    # run_once


if __name__ == "__main__":
    should_stop = Value(c_bool, False)
    consumer_proc = Process(target=consume_ephemeral_data, args=(TEMP_DATA_DIR, "yep_", should_stop))
    consumer_proc.start()

    streams = streamlink.streams(f"https://twitch.tv/{CHANNEL_NAME}")
    if "audio_only" not in streams:
        sys.exit(0)

    # ffmpeg_proc = ffmpeg.input("pipe:")
    # ffmpeg.run()
    # ffmpeg_proc.run()
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg",
         "-i", "pipe:0",
         "-ac", "1",
         "-ar", "16k",
         "-f", "segment", "-segment_time", "30",
         os.path.join(TEMP_DATA_DIR, "yep_%03d.wav")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    pbar = tqdm(unit_scale=True, unit_divisor=1024, unit="B")
    audio_stream_fd = streams["audio_only"].open()

    for data in iter(partial(audio_stream_fd.read, 16 * 1024), b''):
        ffmpeg_proc.stdin.write(data)

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    # what we want:
    # streamlink twitch.tv/dios_dong audio_only -O | \
    # ffmpeg -i pipe: -ac 1 -ar 16000 -f segment -segment_time 10 /tmp/yep__%05d.wav
