from .tts import TTSModel, OUTPUT_SAMPLE_RATE
import argparse
from pathlib import Path
import wave


def synthesize_corpus(
    corpus_path=Path("corpus.txt"),
    tacotron_path=Path("/path/to/tacotron.pt"),
    waveglow_path=Path("/path/to/waveglow.pt"),
    output_dir=Path("./out_dir"),
):
    tts_model = TTSModel(str(tacotron_path), str(waveglow_path))
    output_dir.mkdir(exist_ok=True)
    for (i, line) in enumerate(open(str(corpus_path)).readlines()):
        print(f'synthesizing... "{line.strip()}"')
        data = tts_model.synth_speech(line.strip())
        out_file = str(output_dir / Path(str(i) + ".wav"))
        with wave.open(out_file, "w") as out_file_h:
            out_file_h.setnchannels(1)  # mono
            out_file_h.setsampwidth(2)  # pcm int16 2bytes
            out_file_h.setframerate(OUTPUT_SAMPLE_RATE)
            out_file_h.writeframes(data)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--tacotron_path",
        type=Path,
        default="./tacotron.pt",
        help="Path to a tacotron2 model",
    )
    parser.add_argument(
        "-w",
        "--waveglow_path",
        type=Path,
        default="./waveglow_256channels.pt",
        help="Path to a waveglow model",
    )
    parser.add_argument(
        "-c",
        "--corpus_path",
        type=Path,
        default="./corpus.txt",
        help="Path to a corpus file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default="./synth",
        help="Path to a output directory",
    )
    args = parser.parse_args()
    synthesize_corpus(**vars(args))


if __name__ == "__main__":
    main()
