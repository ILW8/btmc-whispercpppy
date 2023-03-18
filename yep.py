"""
Demonstrate how to use the C++ binding directly.
"""
import collections.abc
import datetime
# from __future__ import annotations

import math
import sys
import functools as f
import time
from os import environ
from pathlib import Path

import whispercpp as w

import tqdm
from typing.io import IO

tqdm.tqdm.format_sizeof_original = tqdm.tqdm.format_sizeof


def format_sizeof(num, suffix='', divisor=-727):
    if divisor != -727:
        return tqdm.tqdm.format_sizeof_original(num, suffix, divisor)

    # if divisor is -727, format num as a timestamp
    return to_timestamp(int(num)) + suffix


tqdm.tqdm.format_sizeof = format_sizeof

_model: w.Whisper | None = None

_MODEL_NAME = environ.get("GGML_MODEL", "medium.en")
_MAX_CONTEXT = int(environ.get("MAX_CONTEXT", "16384"))

k_colors = [
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
]

srt_colors = [
    "#ff6440",
    "#ff752b",
    "#fd870b",
    "#f59a00",
    "#eaac00",
    "#dabe00",
    "#c6d000",
    "#ace000",
    "#8bf000",
    "#56ff40",
]


@f.lru_cache(maxsize=1)
def get_model() -> w.Whisper:
    global _model
    if _model is None:
        _model = w.Whisper.from_pretrained(_MODEL_NAME)
        print(f"Capabilities: {_model.context.sys_info()}", file=sys.stderr)
    return _model


def to_timestamp_from_ms(time_in_ms: int) -> str:
    return to_timestamp(time_in_ms // 10)


def to_timestamp(time_in_10ms: int, comma: bool = False) -> str:
    """
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
    """
    msec: int = time_in_10ms * 10
    hours: int = msec // (1000 * 60 * 60)
    msec = msec - hours * 1000 * 60 * 60
    minutes: int = msec // (1000 * 60)
    msec = msec - minutes * 1000 * 60
    seconds: int = msec // 1000
    msec = msec - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{',' if comma else '.'}{msec:03d}"


def multi_callback_entrypoint(ctx: w.api.Context, n_new: int, userdata: dict) -> None:
    # expected format: userdata: {"use_colors": [], "output_file": []}
    # each callback gets its parameter using their order in this function

    # 0
    print_callback(ctx, n_new, userdata | {"use_colors":  userdata["use_colors"][0],
                                           "output_file": userdata["output_file"][0]})

    # 1
    save_to_srt(ctx, n_new, userdata | {"use_colors":  userdata["use_colors"][1],
                                        "output_file": userdata["output_file"][1]})

    # 2
    print_callback(ctx, n_new, userdata | {"use_colors":  userdata["use_colors"][2],
                                           "output_file": userdata["output_file"][2]})


def print_callback(ctx: w.api.Context, n_new: int, userdata: dict):
    params: w.api.Params = userdata["params"]
    out_file: IO = userdata.get("output_file", sys.stdout)
    use_colors: bool = userdata.get("use_colors", False)

    n_segments = ctx.full_n_segments()

    for i in range(n_segments - n_new, n_segments):
        print("\r" if out_file.isatty() else "", end="", file=out_file)
        if params.print_timestamps:
            print(f"[{to_timestamp(ctx.full_get_segment_start(i))} --> {to_timestamp(ctx.full_get_segment_end(i))}]  ",
                  end="", file=out_file)

        for j in range(ctx.full_n_tokens(i)):
            if not params.print_special:
                w_token_id = ctx.full_get_token_id(i, j)
                if w_token_id >= ctx.eot_token:
                    continue

            try:
                text = ctx.full_get_token_text(i, j)
            except UnicodeDecodeError as e:
                print(f"Unable to parse token text: {e}", file=sys.stderr)
                text = "?TokenDecodeFailed?"
            proba = ctx.full_get_token_prob(i, j)

            if use_colors:
                color = max(0, min(len(k_colors) - 1, int(math.pow(proba, 3) * len(k_colors))))
                print(k_colors[color], end="", file=out_file)
                print(text, end="\033[0m", file=out_file)
                continue

            print(text, end="", file=out_file)
        print("", file=out_file)
    return


def save_to_srt(ctx: w.api.Context, n_new: int, userdata: dict):
    params: w.api.Params = userdata["params"]
    out_file: IO = userdata["output_file"]
    use_colors: bool = userdata.get("use_colors", False)

    n_segments = ctx.full_n_segments()

    for i in range(n_segments - n_new, n_segments):
        out_file.write(f"{i}\n")

        out_file.write(f"{to_timestamp(ctx.full_get_segment_start(i), True)} --> "
                       f"{to_timestamp(ctx.full_get_segment_end(i), True)}\n")

        for j in range(ctx.full_n_tokens(i)):
            if not params.print_special:
                w_token_id = ctx.full_get_token_id(i, j)
                if w_token_id >= ctx.eot_token:
                    continue

            text = ctx.full_get_token_text(i, j)

            out_file.write(f"<font color={srt_colors[get_color_index(srt_colors, ctx.full_get_token_prob(i, j))]}>"
                           if use_colors else "")
            out_file.write(text)
            out_file.write("</font>" if use_colors else "")
        out_file.write("\n\n")
    return


def get_color_index(colors: collections.abc.Sized, token_proba: float):
    return max(0, min(len(colors) - 1, int(math.pow(token_proba, 3) * len(colors))))


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        sys.stderr.write("Usage: yep.py <audio file> [audio files...]\n")
        sys.stderr.flush()
        return 1

    if len(argv) == 1:
        path = argv.pop(0)
        assert Path(path).exists()

        # pbar = tqdm.tqdm()
        run_once(path, multi_callback_entrypoint, True, str(datetime.datetime.utcnow().timestamp()))

        return 0

    for path in argv:
        plp = Path(path)
        assert plp.exists()
        run_once(path, multi_callback_entrypoint, True, f"{plp.stem}_{_MODEL_NAME}_{_MAX_CONTEXT}")


def run_once(file_path, on_new_segment, print_inference_time=False, output_file_name: str = ""):
    _model_was_none = _model is None
    params = get_model().params.build()
    assert _model is not None

    if len(output_file_name) == 0 or Path(f"{output_file_name}.srt").exists() or Path(f"{output_file_name}.txt").exists():
        raise RuntimeError("invalid filename/files exist, not overwriting")

    srt_out = open(f"{output_file_name}.srt", "w")
    txt_out = open(f"{output_file_name}.txt", "w")

    params.on_new_segment(on_new_segment, {"params":      params,
                                           "output_file": [txt_out, srt_out, sys.stdout],
                                           "use_colors":  [False, True, True]})

    params.with_entropy_thold(2.53)
    params.with_num_threads(3)
    params.with_speed_up(False)
    # strategies: w.api.SamplingStrategies = w.api.SamplingStrategies.from_enum(w.api.SAMPLING_BEAM_SEARCH)
    # strategies.beam_search.with_beam_size(3)
    # params.from_sampling_strategy(strategies)

    params.with_num_max_text_ctx(_MAX_CONTEXT)
    # params.with_offset_ms(180000)
    # params.with_duration_ms(120000)

    # start_time = time.perf_counter()
    audio_data = w.api.load_wav_file(Path(file_path).__fspath__()).mono
    _model.context.full(params, audio_data)

    # print(f"Inference time: {time.perf_counter() - start_time:.03f}s\n" if print_inference_time else "", end="")
    if print_inference_time:
        _model.context.print_timings()

    srt_out.close()
    txt_out.close()

    # time.sleep(0.25)
    # pbar.update(pbar.total)
    # pbar.refresh()
    # pbar.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
