# AudioLLMs

## Table of Contents
- [Models and Methods](#models-and-methods)
- [Evaluation](#evaluation)
- [Survey Papers](#survey-papers)
- [Multimodality Models (language + audio + other modalities)](#multimodality-models-language--audio--other-modalities)
- [Adversarial Attacks](#adversarial-attacks)

## Introduction
This repository is a curated collection of research papers focused on the development, implementation, and evaluation of language models for audio data. Our goal is to provide researchers and practitioners with a comprehensive resource to explore the latest advancements in AudioLLMs. Contributions and suggestions for new papers are highly encouraged!

---
---

## Models and Methods
```
【Date】【Name】【Affiliations】【Paper】【Link】
```
- `【2024-12】-【MERaLiON-AudioLLM】-【I2R, A*STAR, Singapore】`
  - MERaLiON-AudioLLM: Bridging Audio and Language with Large Language Models
  - [Paper](https://arxiv.org/abs/2412.09818) / [HF Model](https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION)

- `【2024-09】-【MoWE-Audio】-【A*STAR】`
  - MoWE-Audio: Multitask AudioLLMs with Mixture of Weak Encoders
  - [Paper](https://www.arxiv.org/pdf/2409.06635)

- `【2024-11】-【NTU, Taiwan】`
  - Building a Taiwanese Mandarin Spoken Language Model: A First Attempt
  - [Paper](https://arxiv.org/pdf/2411.07111)

- `【2024-10】-【SPIRIT LM】-【Meta】`
  - SPIRIT LM: Interleaved Spoken and Written Language Model
  - [Paper](https://arxiv.org/pdf/2402.05755) / [Project](https://speechbot.github.io/spiritlm/) / [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/spiritlm?style=social)](https://github.com/facebookresearch/spiritlm) 

- `【2024-10】-【DiVA】-【Georgia Tech, Stanford】`
  - Distilling an End-to-End Voice Assistant Without Instruction Training Data
  - [Paper](https://arxiv.org/pdf/2410.02678) / [Project](https://diva-audio.github.io/)

- `【2024-10】-【SpeechEmotionLlama】-【MIT, Meta】`
  - Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech
  - [Paper](https://arxiv.org/pdf/2410.01162)
 
- `【2024-09】-【Moshi】-【Kyutai】`
  - Developing Instruction-Following Speech Language Model Without Speech Instruction-Tuning Data
  - [Paper](https://arxiv.org/pdf/2409.20007) / [![GitHub stars](https://img.shields.io/github/stars/kyutai-labs/moshi?style=social)](https://github.com/kyutai-labs/moshi)

- `【2024-09】-【DeSTA2】-【NTU Taiwan】`
  - Moshi: a speech-text foundation model for real-time dialogue
  - [Paper](https://arxiv.org/pdf/2410.00037) / [![GitHub stars](https://img.shields.io/github/stars/kehanlu/DeSTA2?style=social)](https://github.com/kehanlu/DeSTA2)

- `【2024-09】-【LLaMA-Omni】-【CAS】`
  - LLaMA-Omni: Seamless Speech Interaction with Large Language Models
  - [Paper](https://arxiv.org/pdf/2409.06666v1) / [![GitHub stars](https://img.shields.io/github/stars/ictnlp/llama-omni?style=social)](https://github.com/ictnlp/llama-omni)

- `【2024-09】-【Ultravox】-【fixie-ai】`
  - GitHub Open Source
  - [![GitHub stars](https://img.shields.io/github/stars/fixie-ai/ultravox?style=social)](https://github.com/fixie-ai/ultravox)

- `【2024-09】-【AudioBERT】-【Postech】`
  - AudioBERT: Audio Knowledge Augmented Language Model
  - [Paper](https://arxiv.org/pdf/2409.08199) / [![GitHub stars](https://img.shields.io/github/stars/HJ-Ok/AudioBERT?style=social)](https://github.com/HJ-Ok/AudioBERT)
 
- `【2024-09】-【-】-【Tsinghua SIGS】`
  - Comparing Discrete and Continuous Space LLMs for Speech Recognition
  - [Paper](https://arxiv.org/pdf/2409.00800v1)

- `【2024-08】-【Mini-Omni】-【Tsinghua】`
  - Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming
  - [Paper](https://arxiv.org/pdf/2408.16725) / [![GitHub stars](https://img.shields.io/github/stars/gpt-omni/mini-omni?style=social)](https://github.com/gpt-omni/mini-omni)

- `【2024-08】-【MooER】-【Moore Threads】`
  - MooER: LLM-based Speech Recognition and Translation Models from Moore Threads
  - [Paper](https://arxiv.org/pdf/2408.05101) / [![GitHub stars](https://img.shields.io/github/stars/MooreThreads/MooER?style=social)](https://github.com/MooreThreads/MooER)

- `【2024-07】-【GAMA】-【UMD】`
  - GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities
  - [Paper](https://arxiv.org/abs/2406.11768) / [![GitHub stars](https://img.shields.io/github/stars/sreyan88/gamaaudio?style=social)](https://sreyan88.github.io/gamaaudio/)

- `【2024-07】-【LLaST】-【CUHK-SZ】`
  - LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models
  - [Paper](https://arxiv.org/pdf/2407.15415) / [![GitHub stars](https://img.shields.io/github/stars/openaudiolab/LLaST?style=social)](https://github.com/openaudiolab/LLaST)

- `【2024-07】-【CompA】-【University of Maryland】`
  - CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models
  - [Paper](https://arxiv.org/abs/2310.08753) / [![GitHub stars](https://img.shields.io/github/stars/Sreyan88/CompA?style=social)](https://github.com/Sreyan88/CompA) / [Project](https://sreyan88.github.io/compa_iclr/)

- `【2024-07】-【Qwen2-Audio】-【Alibaba】`
  - Qwen2-Audio Technical Report
  - [Paper](https://arxiv.org/abs/2407.10759) / [![GitHub stars](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio?style=social)](https://github.com/QwenLM/Qwen2-Audio)

- `【2024-07】-【FunAudioLLM】-【Alibaba】`
  - FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs
  - [Paper](https://arxiv.org/pdf/2407.04051v3) / [![GitHub stars](https://img.shields.io/github/stars/FunAudioLLM?style=social)](https://github.com/FunAudioLLM) / [Demo](https://fun-audio-llm.github.io/)

- `【2024-07】-【-】-【NTU-Taiwan, Meta】`
  - Investigating Decoder-only Large Language Models for Speech-to-text Translation
  - [Paper](https://arxiv.org/pdf/2407.03169)

- `【2024-06】-【Speech ReaLLM】-【Meta】`
  - Speech ReaLLM – Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time
  - [Paper](https://arxiv.org/pdf/2406.09569)

- `【2024-06】-【DeSTA】-【NTU-Taiwan, Nvidia】`
  - DeSTA: Enhancing Speech Language Models through Descriptive Speech-Text Alignment
  - [Paper](https://arxiv.org/abs/2406.18871) / [![GitHub stars](https://img.shields.io/github/stars/kehanlu/Nemo?style=social)](https://github.com/kehanlu/Nemo/tree/desta/examples/multimodal/DeSTA)

- `【2024-05】-【Audio Flamingo】-【Nvidia】`
  - Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities
  - [Paper](https://arxiv.org/abs/2402.01831) / [![GitHub stars](https://img.shields.io/github/stars/NVIDIA/audio-flamingo?style=social)](https://github.com/NVIDIA/audio-flamingo)

- `【2024-04】-【SALMONN】-【Tsinghua】`
  - SALMONN: Towards Generic Hearing Abilities for Large Language Models
  - [Paper](https://arxiv.org/pdf/2310.13289.pdf) / [![GitHub stars](https://img.shields.io/github/stars/bytedance/SALMONN?style=social)](https://github.com/bytedance/SALMONN) / [Demo](https://huggingface.co/spaces/tsinghua-ee/SALMONN-7B-gradio)

- `【2024-03】-【WavLLM】-【CUHK】`
  - WavLLM: Towards Robust and Adaptive Speech Large Language Model
  - [Paper](https://arxiv.org/pdf/2404.00656) / [![GitHub stars](https://img.shields.io/github/stars/microsoft/SpeechT5?style=social)](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)

- `【2024-02】-【SLAM-LLM】-【SJTU】`
  - An Embarrassingly Simple Approach for LLM with Strong ASR Capacity
  - [Paper](https://arxiv.org/pdf/2402.08846) / [![GitHub stars](https://img.shields.io/github/stars/X-LANCE/SLAM-LLM?style=social)](https://github.com/X-LANCE/SLAM-LLM)

- `【2024-01】-【Pengi】-【Microsoft】`
  - Pengi: An Audio Language Model for Audio Tasks
  - [Paper](https://arxiv.org/pdf/2305.11834.pdf) / [![GitHub stars](https://img.shields.io/github/stars/microsoft/Pengi?style=social)](https://github.com/microsoft/Pengi)

- `【2023-12】-【Qwen-Audio】-【Alibaba】`
  - Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models
  - [Paper](https://arxiv.org/pdf/2311.07919.pdf) / [![GitHub stars](https://img.shields.io/github/stars/QwenLM/Qwen-Audio?style=social)](https://github.com/QwenLM/Qwen-Audio) / [Demo](https://qwen-audio.github.io/Qwen-Audio/)

- `【2023-10】-【UniAudio】-【CUHK】`
  - An Audio Foundation Model Toward Universal Audio Generation
  - [Paper](https://arxiv.org/abs/2310.00704) / [![GitHub stars](https://img.shields.io/github/stars/yangdongchao/UniAudio?style=social)](https://github.com/yangdongchao/UniAudio) / [Demo](https://dongchaoyang.top/UniAudio_demo/)

- `【2023-09】-【LLaSM】-【LinkSoul.AI】`
  - LLaSM: Large Language and Speech Model
  - [Paper](https://arxiv.org/pdf/2308.15930.pdf) / [![GitHub stars](https://img.shields.io/github/stars/LinkSoul-AI/LLaSM?style=social)](https://github.com/LinkSoul-AI/LLaSM)

- `【2023-09】-【Segment-level Q-Former】-【Tsinghua】`
  - Connecting Speech Encoder and Large Language Model for ASR
  - [Paper](https://arxiv.org/pdf/2309.13963)

- `【2023-07】-【-】-【Meta】`
  - Prompting Large Language Models with Speech Recognition Abilities
  - [Paper](https://arxiv.org/pdf/2307.11795)

- `【2023-05】-【SpeechGPT】-【Fudan】`
  - SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities
  - [Paper](https://arxiv.org/pdf/2305.11000.pdf) / [![GitHub stars](https://img.shields.io/github/stars/0nutation/SpeechGPT?style=social)](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) / [Demo](https://0nutation.github.io/SpeechGPT.github.io/)

- `【2023-04】-【AudioGPT】-【Zhejiang Uni】`
  - AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head
  - [Paper](https://arxiv.org/pdf/2304.12995.pdf) / [![GitHub stars](https://img.shields.io/github/stars/AIGC-Audio/AudioGPT?style=social)](https://github.com/AIGC-Audio/AudioGPT)

---
---
## Evaluation

```
【Date】【Name】【Affiliations】【Paper】【Link】
```

- `【2024-06】-【AudioBench】-【A*STAR, Singapore】`
  - AudioBench: A Universal Benchmark for Audio Large Language Models
  - [Paper](https://arxiv.org/abs/2406.16020) / [LeaderBoard](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard) / [![GitHub stars](https://img.shields.io/github/stars/AudioLLMs/AudioBench?style=social)](https://github.com/AudioLLMs/AudioBench)

- `【2024-12】-【ADU-Bench】-【Tsinghua, Oxford】`
  - Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models
  - [Paper](https://arxiv.org/abs/2412.05167)

- `【2024-10】-【VoiceBench】-【NUS】`
  - VoiceBench: Benchmarking LLM-Based Voice Assistants
  - [Paper](https://arxiv.org/pdf/2410.17196) / [![GitHub stars](https://img.shields.io/github/stars/MatthewCYM/VoiceBench?style=social)](https://github.com/MatthewCYM/VoiceBench)

- `【2024-09】-【Salmon】-【Hebrew University of Jerusalem】`
  - A Suite for Acoustic Language Model Evaluation
  - [Paper](https://arxiv.org/abs/2409.07437) / [Code](https://pages.cs.huji.ac.il/adiyoss-lab/salmon/)

- `【2024-07】-【AudioEntailment】-【CMU, Microsoft】`
  - Audio Entailment: Assessing Deductive Reasoning for Audio Understanding
  - [Paper](https://arxiv.org/pdf/2407.18062) / [![GitHub stars](https://img.shields.io/github/stars/microsoft/AudioEntailment?style=social)](https://github.com/microsoft/AudioEntailment)

- `【2024-06】-【SD-Eval】-【CUHK, Bytedance】`
  - SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words
  - [Paper](https://arxiv.org/pdf/2406.13340) / [![GitHub stars](https://img.shields.io/github/stars/amphionspace/SD-Eval?style=social)](https://github.com/amphionspace/SD-Eval)

- `【2024-06】-【Audio Hallucination】-【NTU-Taiwan】`
  - Understanding Sounds, Missing the Questions: The Challenge of Object Hallucination in Large Audio-Language Models
  - [Paper](https://arxiv.org/pdf/2406.08402) / [![GitHub stars](https://img.shields.io/github/stars/kuan2jiu99/audio-hallucination?style=social)](https://github.com/kuan2jiu99/audio-hallucination)

- `【2024-05】-【AIR-Bench】-【ZJU, Alibaba】`
  - AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension
  - [Paper](https://aclanthology.org/2024.acl-long.109/) / [![GitHub stars](https://img.shields.io/github/stars/OFA-Sys/AIR-Bench?style=social)](https://github.com/OFA-Sys/AIR-Bench)

- `【2024-08】-【MuChoMusic】-【UPF, QMUL, UMG】`
  - MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models
  - [Paper](https://arxiv.org/abs/2408.01337) / [![GitHub stars](https://img.shields.io/github/stars/mulab-mir/muchomusic?style=social)](https://github.com/mulab-mir/muchomusic)

- `【2023-09】-【Dynamic-SUPERB】-【NTU-Taiwan, etc.】`
  - Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech
  - [Paper](https://arxiv.org/abs/2309.09510) / [![GitHub stars](https://img.shields.io/github/stars/dynamic-superb/dynamic-superb?style=social)](https://github.com/dynamic-superb/dynamic-superb)

---
---
## Survey Papers

- `【2024-11】-【Zhejiang University】`
  - WavChat: A Survey of Spoken Dialogue Models
  - [Paper](https://arxiv.org/abs/2411.13577)

- `【2024-10】-【CUHK, Tencent】`
  - Recent Advances in Speech Language Models: A Survey
  - [Paper](https://arxiv.org/pdf/2410.03751)

- `【2024-10】-【SJTU, AISpeech】`
  - A Survey on Speech Large Language Models
  - [Paper](https://arxiv.org/pdf/2410.18908v2)

---
---
## Multimodality Models (language + audio + other modalities)

To list out some multimodal models that could process audio (speech, non-speech, music, audio-scene, sound, etc.) and text inputs.

|  Date   |       Model          |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-09 |     EMOVA            |      HKUST             | EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions | [Paper](https://arxiv.org/pdf/2409.18042) / [Demo](https://emova-ollm.github.io/) |
| 2023-11 |     CoDi-2           |      UC Berkeley       | CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation | [Paper](https://arxiv.org/pdf/2311.18775) / [Code](https://github.com/microsoft/i-Code/tree/main/CoDi-2) / [Demo](https://codi-2.github.io/) |
| 2023-06 |     Macaw-LLM        |      Tencent           | Macaw-LLM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration | [Paper](https://arxiv.org/pdf/2306.09093) / [Code](https://github.com/lyuchenyang/Macaw-LLM) |

---
---
## Adversarial Attacks

|  Date   |       Name           |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-05 |     VoiceJailbreak   |      CISPA             | Voice Jailbreak Attacks Against GPT-4o | [Paper](https://arxiv.org/pdf/2405.19103) |



#### TODO
- Update the table to text format to allow more structured display

