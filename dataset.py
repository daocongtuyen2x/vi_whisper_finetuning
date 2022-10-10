import os
import numpy as np

import torch
import torchaudio

import pandas as pd
import whisper
import torchaudio.transforms as at
from utils import load_wave
from pathlib import Path
 
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, texts = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            texts.append(f["text"])
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        batch["texts"] = texts
        return batch


#------------------------------------FLUERS------------------------------------#

def get_list_files(phase,audio_path = 'fluers/vi_vn/audio', text_max_length=1000, audio_max_sample_length=960000, sample_rate=16000):
    audio_path = os.path.join(audio_path, phase)
    audio_transcript_pair_list = []
    if phase=='train':
        tsv_file = 'fluers/vi_vn/train.tsv'
    elif phase=='dev':
        tsv_file = 'fluers/vi_vn/dev.tsv'
    else:
        tsv_file = 'fluers/vi_vn/test.tsv'
    df = pd.read_table(tsv_file, names=("id", "file_name", "raw_transcription", "transcription", "_", "num_samples", "gender"))
    for index, row in df.iterrows():
        new_path = Path(os.path.join(audio_path, row['file_name']))
        audio_id = row['id']
        text = row['transcription']
        if new_path.exists():
            audio = load_wave(new_path, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print('skip file:', new_path,'with len text:', len(text), 'and len audio', len(audio))
                continue
            audio_transcript_pair_list.append((audio_id, str(new_path), text))
    return audio_transcript_pair_list


class VNFluers(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_rate=16000) -> None:
        super().__init__()

        self.dataset = dataset
        self.sample_rate = sample_rate

        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform
    

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.dataset[id]

        audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text_token = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        labels = text_token[1:] + [self.tokenizer.eot]
        if len(text_token) >= 448:
            audio_id, audio_path, text = self.dataset[0]

            audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
            audio = whisper.pad_or_trim(audio.flatten())
            mel = whisper.log_mel_spectrogram(audio)

            text_token = [
                *self.tokenizer.sot_sequence_including_notimestamps
            ] + self.tokenizer.encode(text)
            labels = text_token[1:] + [self.tokenizer.eot]
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }


def load_fluers():
    print('Loading Vietnamese Fluers dataset...')
    os.system("wget https://storage.googleapis.com/xtreme_translations/FLEURS102/vi_vn.tar.gz")
    os.makedirs('fluers', exist_ok=True)
    os.system("tar -xvf 'vi_vn.tar.gz' -C fluers")


    train_list_files = get_list_files('train')
    val_list_files = get_list_files('dev')
    test_list_files = get_list_files('test')
    train_list_files +=val_list_files
    print('Num train samples:', len(train_list_files))
    print('Num test samples:', len(test_list_files))

    train_dataset = VNFluers(train_list_files)
    test_dataset = VNFluers(test_list_files)
    return train_dataset, test_dataset
if __name__=='__main__':
    load_fluers()




