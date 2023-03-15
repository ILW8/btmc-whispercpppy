import streamlink
import ffmpeg  # python-ffmpeg


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


if __name__ == "__main__":
    a = streamlink.Streamlink
    b = ffmpeg

    # what we want:
    # streamlink twitch.tv/dios_dong audio_only -O | \
    # ffmpeg -i pipe: -ac 1 -ar 16000 -f segment -segment_time 10 /tmp/yep__%05d.wav
