# Anonymous Submission for [Interspeech 2025]

This repository contains the source code for our submission to Interspeech 2025. The implementation is anonymized to comply with the double-blind review process.

## Overview

This work focuses on DSR (Dysarthric Speech Recognition). We provide the training and inference scripts, along with instructions for reproducibility.

## Dependencies

Ensure you have the following dependencies installed:

```
pip install -r requirements.txt
```

## Training

To train the model, use the following command:
```
mv dataset ./whisper-finetune
cd whisper-finetune
python finetune.py # for Single-GPU
python torchrun --nproc_per_node=4 finetune.py # for Multi-GPU
```

## Merge model

```
python merge_lora.py
```

## Ctranslate2 Inference

To run inference faster on a sample:
```
ct2-transformers-converter --model models/whisper-large-v2-finetune --output_dir models/whisper-large-v2-finetune-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
python infer_ct2.py --input example.wav --output output.wav
```

## Submissions Results

Results of Submissions from the witty Team

| **Method**                 | **Test 1 WER** | **Test 1 SemScore** | **Submitted at** |
|----------------------------|---------------|---------------------|----------------|
| step0 finetune             | 10.02         | 87.38               | 2025-01-13     |
| +deduplication             | 9.92          | 87.99               | 2025-01-19     |
| +increase volume           | 8.75          | 88.81               | 2025-01-24     |
| >norm volume               | 8.69          | 88.93               | 2025-01-25     |
| all norm volume            | 8.48          | 89.38               | 2025-01-27     |
| step1 finetune             | 8.13          | 89.88               | 2025-01-31     |
| +error correlation         | 8.09          | 90.17               | 2025-02-01     |
| step2 finetune             | 7.98          | 90.37               | 2025-02-02     |

## Contact

For questions, please refer to the conference submission system for anonymous discussions.
