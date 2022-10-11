import whisper
import torch
import argparse

from config import Config
from dataset import load_fluers, WhisperDataCollatorWithPadding
from model import WhisperModelModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='', help='path of checkpoint, if not set, use origin pretrained model')

    args = parser.parse_args()
    config = Config()
    config.checkpoint_path = args.checkpoint_path

    module = WhisperModelModule(config)
    try:
        state_dict = torch.load(config.checkpoint_path)
        state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict)
        print(f"load checkpoint successfully from {config.checkpoint_path}")
    except Exception as e:
        print(e)
        print(f"load checkpoint failt using origin weigth of {config.model_name} model")
    model = module.model
    model.to(device)

    _, valid_dataset = load_fluers()
    test_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        collate_fn=WhisperDataCollatorWithPadding(),
    )

    # decode the audio
    options = whisper.DecodingOptions(
    language="vi", without_timestamps=True, fp16=torch.cuda.is_available()
    )

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            print('label:', batch["labels"])
            result = whisper.decode(model, input_ids, options)
            print('predict:', result.text, '\n')

