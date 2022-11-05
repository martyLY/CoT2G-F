import datasets
from transformers import T5Config
from t5_tokenizer_model import SentencePieceUnigramTokenizer

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='the dictionary you save model config and tokenizer.')
    parser.add_argument('--protein-file-name', type=str, default=None,
                        help='protein sequence file name, need to be txt.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    vocab_size = 32_000
    input_sentence_size = None
    # Initialize a dataset

    args = parse_args()
    dataset = datasets.load_dataset("text", data_files={"train": args.protein_file_name})

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 100
        for i in range(0, input_sentence_size, batch_length):
            yield dataset["train"][i: i + batch_length]["text"]


    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save("./{}/tokenizer.json".format(args.model_dir))

    config = T5Config.from_pretrained("t5-small", vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained("./{}".format(args.model_dir))


if __name__ == "__main__":
    main()