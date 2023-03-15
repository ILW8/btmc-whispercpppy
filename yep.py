"""
Demonstrate how to use the C++ binding directly.
"""
# from __future__ import annotations

import math
import sys
import functools as f
import time
from os import environ
from pathlib import Path

import whispercpp as w

import tqdm


tqdm.tqdm.format_sizeof_original = tqdm.tqdm.format_sizeof


def format_sizeof(num, suffix='', divisor=-727):
    if divisor != -727:
        return tqdm.tqdm.format_sizeof_original(num, suffix, divisor)

    # if divisor is -727, format num as a timestamp
    return to_timestamp(int(num)) + suffix


tqdm.tqdm.format_sizeof = format_sizeof

_model: w.Whisper | None = None

_MODEL_NAME = environ.get("GGML_MODEL", "small.en")

k_colors = [
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
]


@f.lru_cache(maxsize=1)
def get_model() -> w.Whisper:
    global _model
    if _model is None:
        _model = w.Whisper.from_pretrained(_MODEL_NAME)
    return _model


def to_timestamp_from_ms(time_in_ms: int) -> str:
    return to_timestamp(time_in_ms//10)


def to_timestamp(time_in_10ms: int) -> str:
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
    hours: int = msec // (1000*60*60)
    msec = msec - hours * 1000 * 60 * 60
    minutes: int = msec // (1000 * 60)
    msec = msec - minutes * 1000 * 60
    seconds: int = msec // 1000
    msec = msec - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{msec:03d}"


def print_callback(ctx: w.api.Context, n_new: int, userdata: dict):
    params: w.api.Params = userdata["params"]
    pbar: tqdm.tqdm = userdata["pbar"]

    if pbar.total is None:
        pbar.total = ctx.n_len
        # pbar.refresh()

    n_segments = ctx.full_n_segments()

    # last = None
    for i in range(n_segments - n_new, n_segments):
        """
            if (!params.no_timestamps) {
                printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
            }
        """
        print("\r", end="")
        if params.print_timestamps:
            print(f"[{to_timestamp(ctx.full_get_segment_start(i))} --> {to_timestamp(ctx.full_get_segment_end(i))}]  ",
                  end="")

        """
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
            if (params.print_special == false) {
                const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                if (id >= whisper_token_eot(ctx)) {
                    continue;
                }
            }
        
            const char * text = whisper_full_get_token_text(ctx, i, j);
            const float  p    = whisper_full_get_token_p   (ctx, i, j);
        
            const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));
            std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size())))
        
            printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
        """
        for j in range(ctx.full_n_tokens(i)):
            if not params.print_special:
                w_token_id = ctx.full_get_token_id(i, j)
                if w_token_id >= ctx.eot_token:
                    continue

            text = ctx.full_get_token_text(i, j)
            proba = ctx.full_get_token_prob(i, j)

            color = max(0, min(len(k_colors) - 1, int(math.pow(proba, 3) * len(k_colors))))
            print(k_colors[color], end="")
            print(text, end="\033[0m")
        print()
        # last = i
    # if last is not None:
    #     pbar.update(ctx.full_get_segment_end(last) - pbar.n)
    return


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        sys.stderr.write("Usage: yep.py <audio file>\n")
        sys.stderr.flush()
        return 1

    path = argv.pop(0)
    assert Path(path).exists()

    # pbar = tqdm.tqdm()
    params = get_model().params.build()
    pbar = tqdm.tqdm(unit_scale=True, unit_divisor=-727, smoothing=0, unit="")
    params.on_new_segment(print_callback, {"params": params, "pbar": pbar})
    assert _model is not None
    _model.context.full(params, w.api.load_wav_file(Path(path).__fspath__()).mono)

    time.sleep(0.25)
    pbar.update(pbar.total)
    pbar.refresh()
    pbar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

