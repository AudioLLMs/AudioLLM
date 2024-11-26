# AudioLLMs

This repository is a curated collection of research papers focused on the development, implementation, and evaluation of language models for audio data. Our goal is to provide researchers and practitioners with a comprehensive resource to explore the latest advancements in AudioLLMs. Contributions and suggestions for new papers are highly encouraged!

## Models

|  Date   |       Model          |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-10 |     SPIRIT LM        |      Meta              | SPIRIT LM: Interleaved Spoken and Written Language Model | [Paper](https://arxiv.org/pdf/2402.05755) / [Code](https://github.com/facebookresearch/spiritlm) / [Project](https://speechbot.github.io/spiritlm/) |
| 2024-10 |     DiVA             |      Georgia Tech, Stanford  | Distilling an End-to-End Voice Assistant Without Instruction Training Data | [Paper](https://arxiv.org/pdf/2410.02678) / [Project](https://diva-audio.github.io/) |
| 2024-09 |     Moshi            |      Kyutai            | Moshi: a speech-text foundation model for real-time dialogue | [Paper](https://arxiv.org/pdf/2410.00037) / [Code](https://github.com/kyutai-labs/moshi) |
| 2024-09 |     LLaMA-Omni       |      CAS               | LLaMA-Omni: Seamless Speech Interaction with Large Language Models | [Paper](https://arxiv.org/pdf/2409.06666v1) / [Code](https://github.com/ictnlp/llama-omni) |
| 2024-09 |     Ultravox         |      fixie-ai          | GitHub Open Source | [Code](https://github.com/fixie-ai/ultravox) |
| 2024-08 |     Mini-Omni        |      Tsinghua          | Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming | [Paper](https://arxiv.org/pdf/2408.16725) / [Code](https://github.com/gpt-omni/mini-omni) |
| 2024-08 |     Typhoon-Audio    |      Typhoon           | Typhoon-Audio Preview Release | [Page](https://blog.opentyphoon.ai/typhoon-audio-preview-release-6fbb3f938287) |
| 2024-08 |     USDM             |      SNU               | Integrating Paralinguistics in Speech-Empowered Large Language Models for Natural Conversation | [Paper](https://arxiv.org/pdf/2402.05706) |
| 2024-08 |     MooER            |      Moore Threads     | MooER: LLM-based Speech Recognition and Translation Models from Moore Threads | [Paper](https://arxiv.org/pdf/2408.05101) / [Code](https://github.com/MooreThreads/MooER) |
| 2024-07 |     GAMA             |      UMD               | GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities | [Paper](https://arxiv.org/abs/2406.11768) / [Code](https://sreyan88.github.io/gamaaudio/) |
| 2024-07 |     LLaST            |      CUHK-SZ           | LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models | [Paper](https://arxiv.org/pdf/2407.15415) / [Code](https://github.com/openaudiolab/LLaST) |
| 2024-07 |     CompA            |      University of Maryland           | CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models | [Paper](https://arxiv.org/abs/2310.08753) / [Code](https://github.com/Sreyan88/CompA) / [Project](https://sreyan88.github.io/compa_iclr/) |
| 2024-07 |     Qwen2-Audio      |      Alibaba           | Qwen2-Audio Technical Report | [Paper](https://arxiv.org/abs/2407.10759) / [Code](https://github.com/QwenLM/Qwen2-Audio) |
| 2024-07 |     FunAudioLLM      |      Alibaba           | FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs | [Paper](https://arxiv.org/pdf/2407.04051v3) / [Code](https://github.com/FunAudioLLM) / [Demo](https://fun-audio-llm.github.io/)  |
| 2024-06 |     BESTOW           |     NVIDIA             | BESTOW: Efficient and Streamable Speech Language Model with the Best of Two Worlds in GPT and T5 | [Paper](https://arxiv.org/pdf/2406.19954) |
| 2024-06 |     DeSTA            |     NTU-Taiwan, Nvidia | DeSTA: Enhancing Speech Language Models through Descriptive Speech-Text Alignment | [Paper](https://arxiv.org/abs/2406.18871) / [Code](https://github.com/kehanlu/Nemo/tree/desta/examples/multimodal/DeSTA) |
| 2024-05 |     AudioChatLlama   |      Meta              | AudioChatLlama: Towards General-Purpose Speech Abilities for LLMs | [Paper](https://aclanthology.org/2024.naacl-long.309.pdf) |
| 2024-05 |     Audio Flamingo   |      Nvidia            | Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities | [Paper](https://arxiv.org/abs/2402.01831) / [Code](https://github.com/NVIDIA/audio-flamingo) |
| 2024-05 |     SpeechVerse      |      AWS               | SpeechVerse: A Large-scale Generalizable Audio Language Model | [Paper](https://arxiv.org/pdf/2405.08295) |
| 2024-04 |     SALMONN          |      Tsinghua          | SALMONN: Towards Generic Hearing Abilities for Large Language Models | [Paper](https://arxiv.org/pdf/2310.13289.pdf) / [Code](https://github.com/bytedance/SALMONN) / [Demo](https://huggingface.co/spaces/tsinghua-ee/SALMONN-7B-gradio) |
| 2024-03 |     WavLLM           |      CUHK              | WavLLM: Towards Robust and Adaptive Speech Large Language Model | [Paper](https://arxiv.org/pdf/2404.00656) / [Code](https://github.com/microsoft/SpeechT5/tree/main/WavLLM) |
| 2024-02 |     LTU              |      MIT               | Listen, Think, and Understand | [Paper](https://arxiv.org/pdf/2305.10790) / [Code](https://github.com/YuanGongND/ltu) |
| 2024-02 |     SLAM-LLM         |      SJTU              | An Embarrassingly Simple Approach for LLM with Strong ASR Capacity | [Paper](https://arxiv.org/pdf/2402.08846) / [Code](https://github.com/X-LANCE/SLAM-LLM) |
| 2024-01 |     Pengi            |      Microsoft         | Pengi: An Audio Language Model for Audio Tasks | [Paper](https://arxiv.org/pdf/2305.11834.pdf) / [Code](https://github.com/microsoft/Pengi) |
| 2023-12 |     Qwen-Audio       |      Alibaba           | Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models | [Paper](https://arxiv.org/pdf/2311.07919.pdf) / [Code](https://github.com/QwenLM/Qwen-Audio) / [Demo](https://qwen-audio.github.io/Qwen-Audio/) |
| 2023-12 |     LTU-AS           |      MIT               | Joint Audio and Speech Understanding | [Paper](https://arxiv.org/pdf/2309.14405v3.pdf) / [Code](https://github.com/YuanGongND/ltu) / [Demo](https://huggingface.co/spaces/yuangongfdu/ltu-2) |
| 2023-10 |     Speech-LLaMA     |      Microsoft         | On decoder-only architecture for speech-to-text and large language model integration | [Paper](https://arxiv.org/pdf/2307.03917) |
| 2023-10 |     UniAudio         |      CUHK              | An Audio Foundation Model Toward Universal Audio Generation | [Paper](https://arxiv.org/abs/2310.00704) / [Code](https://github.com/yangdongchao/UniAudio) / [Demo](https://dongchaoyang.top/UniAudio_demo/) |
| 2023-09 |     LLaSM            |      LinkSoul.AI       | LLaSM: Large Language and Speech Model | [Paper](https://arxiv.org/pdf/2308.15930.pdf) / [Code](https://github.com/LinkSoul-AI/LLaSM) |
| 2023-06 |     AudioPaLM        |      Google            | AudioPaLM: A Large Language Model that Can Speak and Listen | [Paper](https://arxiv.org/pdf/2306.12925.pdf) / [Demo](https://google-research.github.io/seanet/audiopalm/examples/) |
| 2023-05 |     VioLA            |      Microsoft         | VioLA: Unified Codec Language Models for Speech Recognition, Synthesis, and Translation | [Paper](https://arxiv.org/pdf/2305.16107.pdf) |
| 2023-05 |     SpeechGPT        |      Fudan             | SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities | [Paper](https://arxiv.org/pdf/2305.11000.pdf) / [Code](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) / [Demo](https://0nutation.github.io/SpeechGPT.github.io/) |
| 2023-04 |     AudioGPT         |      Zhejiang Uni      | AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head | [Paper](https://arxiv.org/pdf/2304.12995.pdf) / [Code](https://github.com/AIGC-Audio/AudioGPT) |
| 2022-09 |     AudioLM          |      Google            | AudioLM: a Language Modeling Approach to Audio Generation | [Paper](https://arxiv.org/abs/2209.03143) / [Demo](https://google-research.github.io/seanet/audiolm/examples/) |

## Models (language + audio + other modalities)

|  Date   |       Model          |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-09 |     EMOVA            |      HKUST             | EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions | [Paper](https://arxiv.org/pdf/2409.18042) / [Demo](https://emova-ollm.github.io/) |
| 2023-11 |     CoDi-2           |      UC Berkeley       | CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation | [Paper](https://arxiv.org/pdf/2311.18775) / [Code](https://github.com/microsoft/i-Code/tree/main/CoDi-2) / [Demo](https://codi-2.github.io/) |
| 2023-06 |     Macaw-LLM        |      Tencent           | Macaw-LLM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration | [Paper](https://arxiv.org/pdf/2306.09093) / [Code](https://github.com/lyuchenyang/Macaw-LLM) |

## Methodology

|  Date   |       Name           |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-10 |     SpeechEmotionLlama   |      MIT, Meta         | Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech | [Paper](https://arxiv.org/pdf/2410.01162) |
| 2024-09 |     AudioBERT        |      Postech           | AudioBERT: Audio Knowledge Augmented Language Model | [Paper](https://arxiv.org/pdf/2409.08199) / [Code](https://github.com/HJ-Ok/AudioBERT) |
| 2024-09 |     MoWE-Audio       |      A*STAR            | MoWE-Audio: Multitask AudioLLMs with Mixture of Weak Encoders | [Paper](https://www.arxiv.org/pdf/2409.06635) |
| 2024-09 |     -                |      Tsinghua SIGS     | Comparing Discrete and Continuous Space LLMs for Speech Recognition | [Paper](https://arxiv.org/pdf/2409.00800v1) |
| 2024-07 |     -                |      NTU-Taiwan, Meta  | Investigating Decoder-only Large Language Models for Speech-to-text Translation | [Paper](https://arxiv.org/pdf/2407.03169) |
| 2024-06 |     Speech ReaLLM    |      Meta              | Speech ReaLLM – Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time | [Paper](https://arxiv.org/pdf/2406.09569) |
| 2023-09 |     Segment-level Q-Former      |      Tsinghua      | Connecting Speech Encoder and Large Language Model for ASR | [Paper](https://arxiv.org/pdf/2309.13963) |
| 2023-07 |     -                |      Meta              | Prompting Large Language Models with Speech Recognition Abilities | [Paper](https://arxiv.org/pdf/2307.11795) |

## Adversarial Attacks

|  Date   |       Name           |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-05 |     VoiceJailbreak   |      CISPA             | Voice Jailbreak Attacks Against GPT-4o | [Paper](https://arxiv.org/pdf/2405.19103) |

## Evaluation

|  Date   |       Name           |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-10 |     VoiceBench  |      NUS    | VoiceBench: Benchmarking LLM-Based Voice Assistants | [Paper](https://arxiv.org/pdf/2410.17196) / [Code](https://github.com/MatthewCYM/VoiceBench) |
| 2024-07 |     AudioEntailment  |      CMU, Microsoft    | Audio Entailment: Assessing Deductive Reasoning for Audio Understanding | [Paper](https://arxiv.org/pdf/2407.18062) / [Code](https://github.com/microsoft/AudioEntailment) |
| 2024-06 |     Audio Hallucination  |      NTU-Taiwan    | Understanding Sounds, Missing the Questions: The Challenge of Object Hallucination in Large Audio-Language Models | [Paper](https://arxiv.org/pdf/2406.08402) / [Code](https://github.com/kuan2jiu99/audio-hallucination) |
| 2024-06 |     AudioBench       |      A*STAR, Singapore            | AudioBench: A Universal Benchmark for Audio Large Language Models | [Paper](https://arxiv.org/abs/2406.16020) / [Code](https://github.com/AudioLLMs/AudioBench) / [LeaderBoard](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard)|
| 2024-05 |     AIR-Bench        |      ZJU, Alibaba      | AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension | [Paper](https://aclanthology.org/2024.acl-long.109/) / [Code](https://github.com/OFA-Sys/AIR-Bench) |
| 2024-08 |     MuChoMusic       |      UPF, QMUL, UMG    | MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models | [Paper](https://arxiv.org/abs/2408.01337) / [Code](https://github.com/mulab-mir/muchomusic) |
| 2023-09 |     Dynamic-SUPERB   |      NTU-Taiwan, etc. | Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech | [Paper](https://arxiv.org/abs/2309.09510) / [Code](https://github.com/dynamic-superb/dynamic-superb) |


## Audio Model

Audio Models are different from Audio Large Language Models.

### Evaluation

|  Date   |       Name           |    Key Affiliations    | Paper |    Link     |
| :-----: | :------------------: | :--------------------: | :---- | :---------: |
| 2024-09 |     Salmon       |      Hebrew University of Jerusalem | A Suite for Acoustic Language Model Evaluation | [Paper](https://arxiv.org/abs/2409.07437) / [Code](https://pages.cs.huji.ac.il/adiyoss-lab/salmon/) |


## Survey

|  Date   |    Key Affiliations    | Paper |    Link     |
| :-----: | :--------------------: | :---- | :---------: |
| 2024-11 |      Zhejiang University | WavChat: A Survey of Spoken Dialogue Models | [Paper](https://arxiv.org/abs/2411.13577) |
| 2024-10 |      CUHK, Tencent     | Recent Advances in Speech Language Models: A Survey | [Paper](https://arxiv.org/pdf/2410.03751) |
| 2024-10 |      SJTU, AISpeech    | A Survey on Speech Large Language Models | [Paper](https://arxiv.org/pdf/2410.18908v2) |

