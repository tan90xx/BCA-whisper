#!/usr/bin/env python3
# By xiuwenz2@illinois.edu, July 10, 2024.

"""
SAPC inference file with whisper base model. Please modify it w.r.t. your own model.
"""
import argparse
import os
import re

import numpy as np
import pyloudnorm as pyln
import whisperx
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

# To avoid CPU overload
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# To avoid CPU overload
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-pth", default="/taiga/data/processed", type=str, metavar="DATA_PTH", help="data_pth"
    )
    parser.add_argument(
        "--manifest-pth", default="/taiga/manifest", type=str, metavar="MANIFEST_PTH", help="manifest_pth"
    )
    parser.add_argument(
        "--root", default="/taiga/downloads/???/???", type=str, metavar="ROOT", help="???s represent team name, submission pk."
    )
    parser.add_argument(
        "--split", default="test1", type=str, metavar="SPLIT", help="split"
    )
    parser.add_argument(
        "--output-name", default="/taiga/downloads/???/???/inference/???.hypo", type=str, metavar="OUTPUT-NAME", help="???s represent team name, submission pk, and split."
    )
    return parser

def clean_repeated_words(combined_text, max_word_length=15):
    codes = '''
    \u0041-\u005a\u0061-\u007a\u0030-\u0039\u0027\u0020
    \u00c0\u00c1\u00c4\u00c5\u00c8\u00c9\u00cd\u00cf
    \u00d1\u00d3\u00d6\u00d8\u00db\u00dc
    \u0106
    ''' 
    combined_text = re.sub(u"([^"+codes+"])", " ", combined_text)
    words = combined_text.strip().split()
    cleaned_words = []
    for word in words:
        if len(word) > max_word_length:
            pattern = re.compile(r'(\w+?)\1+')  # 这里修正
            match = pattern.match(word)
            if match:
                word = match.group(1)
            else:
                word = ""
        cleaned_words.append(word)
    combined_text = " ".join(cleaned_words)
    if len(words) > 3:
        combined_text = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', combined_text)
    combined_text = re.sub(r'([A-Z]{2,})', lambda m: ' '.join(m.group(0)), combined_text)
    return combined_text

def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
) -> np.ndarray:

    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    return pyln.normalize.loudness(audio, _loudness, loudness)

def main(args):
    
    ######## TO-DO 1: load your own model ########
    model_path = os.path.join(args.root, "models/whisper-large-v2-finetune-ct2-3350")
    model = whisperx.load_model(model_path, "cuda", compute_type="float16", language="en", local_files_only=True)
    normalizer = Normalizer(input_case='cased', lang='en')
    ##############################################
    ### Dump Inference Results
    manifest = os.path.join(args.manifest_pth, args.split + ".tsv")
    with open(manifest, "r") as ftsv, open(args.output_name, "w") as fhypo:
        next(ftsv)
        for t in tqdm(ftsv.readlines()):
            fname = t.strip().split()[0].split("/")[-1]
            
            ######## TO-DO 2: modify the following inference scripts ########
            audio = whisperx.load_audio(os.path.join(args.data_pth, args.split, fname))
            
            # 处理音频
            try:
                audio = loudness_norm(audio, rate=16000)
            except Exception as e:
                print(f"Warning: Loudness normalization failed for {fname}, using original audio. Error: {e}")

            # 语音转录
            try:
                result = model.transcribe(audio, batch_size=16)
                combined_text = " ".join(item['text'].removeprefix(" ") for item in result['segments'])
            except Exception as e:
                print(f"Error during transcription for {fname}: {e}")
                combined_text = ""

            # 文本后处理
            try:
                output_text = clean_repeated_words(combined_text)
                output_text = normalizer.normalize(output_text, verbose=False, punct_post_process=True)
            except Exception as e:
                print(f"Error during text processing for {fname}: {e}")
                output_text = combined_text  # 失败时，返回原始文本
            #################################################################
            
            print(output_text.strip(), file=fhypo)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
