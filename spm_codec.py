from utils import load_filepaths_and_text

# from text import text_to_sequence, sequence_to_text

from hparams import create_hparams
import sentencepiece as spm
from text import symbols


SPM_CORPUS_FILE = "filelists/text_corpus.txt"
SPM_MODEL_PREFIX = "spm"
SPM_VOCAB_SIZE = 1000
hparams = create_hparams()


def _create_sentencepiece_corpus():
    def get_text_list(text_file):
        return [i[1] + "\n" for i in load_filepaths_and_text(text_file)]

    full_text_list = get_text_list(hparams.training_files) + get_text_list(
        hparams.validation_files
    )
    with open(SPM_CORPUS_FILE, "w") as fd:
        fd.writelines(full_text_list)


def _create_sentencepiece_vocab(vocab_size=SPM_VOCAB_SIZE):
    train_params = "--input={} --model_prefix={} --vocab_size={}".format(
        SPM_CORPUS_FILE, SPM_MODEL_PREFIX, vocab_size
    )
    spm.SentencePieceTrainer.Train(train_params)


def _spm_text_codecs():
    sp = spm.SentencePieceProcessor()
    sp.Load("{}.model".format(SPM_MODEL_PREFIX))

    def ttseq(text, cleaners):
        return sp.EncodeAsIds(text)

    def seqtt(sequence):
        return sp.DecodeIds(sequence)

    return ttseq, seqtt


text_to_sequence, sequence_to_text = _spm_text_codecs()


def _interactive_test():
    prompt = "Hello world; how are you, doing ?"
    while prompt not in ["q", "quit"]:
        oup = sequence_to_text(text_to_sequence(prompt, hparams.text_cleaners))
        print('==> ',oup)
        prompt = input("> ")


def main():
    # _create_sentencepiece_corpus()
    # _create_sentencepiece_vocab()
    _interactive_test()


if __name__ == "__main__":
    main()
