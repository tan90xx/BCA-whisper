# BCA-Whisper

This repository contains the official implementation of our Interspeech 2025 paper: "BCA-Whisper: Curriculum Learning-Based AdaLoRA Fine-Tuning on Whisper for Low-Resource Dysarthric Speech Recognition".

## Overview

This work focuses on DSR (Dysarthric Speech Recognition). We provide the training and inference scripts, along with instructions for reproducibility. For detailed methodology and experiments, please refer to our [solution PDF](https://github.com/tan90xx/BCA-Whisper/blob/main/resources/technical_solution.pdf).

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

## Credits

We gratefully acknowledge the following resources and contributions that made this project possible:<br>
[OpenAI Whisper](https://github.com/openai/whisper) and [WhisperX](https://github.com/m-bain/whisperX): Base speech recognition model architecture.<br>
[Whisper Finetune](https://github.com/yeyupiaoling/Whisper-Finetune): Adaptive Low-Rank Adaptation method for parameter-efficient fine-tuning.<br>
The SAPC Team: For hosting the [competition/evaluation](https://github.com/xiuwenz2/SAPC-template) platform and providing carefully annotated dysarthric speech [dataset](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project).<br>
Anonymous Reviewers: For their constructive feedback.
