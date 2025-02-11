#!/usr/bin/env python3
# By xiuwenz2@illinois.edu, July 10, 2024.

"""
SAPC inference file with whisper base model. Please modify it w.r.t. your own model.
"""

import argparse, os
import re
from tqdm import tqdm
import whisperx
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pyloudnorm as pyln
from nemo_text_processing.text_normalization.normalize import Normalizer

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

def remove_repeated_words(combined_text):
    words = combined_text.strip().split()
    combined_text = ' '.join(words) ### remove extra space
    if len(words) > 3:
        combined_text = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', combined_text)
    combined_text = re.sub(r'([A-Z]{2,})', lambda m: ' '.join(m.group(0)), combined_text)
    return combined_text

def increase_loudness(audio, gain_db):
    gain_linear = 10 ** (gain_db / 20)
    audio_louder = audio * gain_linear
    audio_louder = np.clip(audio_louder, -1.0, 1.0)
    return audio_louder

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
    # model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
    model_path = os.path.join(args.root, "models/whisper-large-v2-finetune-ct2-650")
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
            # audio = increase_loudness(audio, gain_db=5)
            audio = loudness_norm(audio, rate=16000)
            result = model.transcribe(audio, batch_size=16)
            combined_text = " ".join(item['text'].removeprefix(" ") for item in result['segments'])
            output_text = remove_repeated_words(combined_text)
            output_text = normalizer.normalize(output_text, verbose=False, punct_post_process=True)
            #################################################################
            
            print(output_text.strip(), file=fhypo)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
