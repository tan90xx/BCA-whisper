#!/usr/bin/env python3
# By xiuwenz2@illinois.edu, July 10, 2024.

"""
SAPC inference file with whisper base model. Please modify it w.r.t. your own model.
"""

import argparse, os
from tqdm import tqdm
import whisperx
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from openai import OpenAI

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

def main(args):
    def increase_loudness(audio, gain_db):
        gain_linear = 10 ** (gain_db / 20)
        audio_louder = audio * gain_linear
        audio_louder = np.clip(audio_louder, -1.0, 1.0)
        return audio_louder        
    def call_deepseek_api(text):
        client = OpenAI(api_key="sk-616795398fe3485c93cbabc28f005418", base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a speech recognition corrector. Please follow these rules: Fix obvious errors in the spelling of nouns. Delete the repetitions. Do not add any explanations or additional text."},
                    {"role": "user", "content": text},
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return None
    def batch_process_file(input_file, chunk_size=100):
        temp_file = input_file + '.tmp'
        with open(input_file, "r", encoding="utf-8") as infile, open(temp_file, "w", encoding="utf-8") as outfile:
            chunk = []
            for i, line in tqdm(enumerate(infile, 1)):
                chunk.append(line.strip())
                if i % chunk_size == 0:
                    text = "\n".join(chunk)
                    result = call_deepseek_api(text)
                    if result:
                        outfile.write(result + "\n")
                    else:
                        outfile.write(text + "\n")
                    chunk = []
            if chunk:
                text = "\n".join(chunk)
                result = call_deepseek_api(text)
                if result:
                    outfile.write(result + "\n")
        # 替换原文件
        os.replace(temp_file, input_file)
    ######## TO-DO 1: load your own model ########
    # model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
    model_path = os.path.join(args.root, "models/whisper-large-v2-finetune-ct2")
    model = whisperx.load_model(model_path, "cuda", compute_type="float16", language="en", local_files_only=True)
    ##############################################
    ### Dump Inference Results
    manifest = os.path.join(args.manifest_pth, args.split + ".tsv")
    with open(manifest, "r") as ftsv, open(args.output_name, "w") as fhypo:
        next(ftsv)
        for t in tqdm(ftsv.readlines()):
            fname = t.strip().split()[0].split("/")[-1]
            
            ######## TO-DO 2: modify the following inference scripts ########
            audio = whisperx.load_audio(os.path.join(args.data_pth, args.split, fname))
            result = model.transcribe(audio, batch_size=16)
            if not result.get('segments'):
                audio_louder = increase_loudness(audio, 20)
                result = model.transcribe(audio_louder, batch_size=16)
            combined_text = " ".join(item['text'].removeprefix(" ") for item in result['segments'])
            #################################################################
            
            print(combined_text.strip(), file=fhypo)
    ##############################################
    ### Post process
    batch_process_file(fhypo)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
