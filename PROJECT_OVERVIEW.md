# Project Overview

## Summary

This project is a Lithuanian telephony ASR platform designed around a practical problem: call recordings are much harder to transcribe reliably than clean speech data. The repository includes both the serving path for inference and the supporting workflow used to improve the model over time.

The result is an end-to-end system rather than a single-model demo. It includes audio preprocessing, decoding, confidence-aware recovery, language-specific cleanup, API delivery, dataset building, fine-tuning, evaluation, and regression testing in one codebase.

## Why This Is a Pipeline Problem

For this kind of audio, model quality depends on more than the checkpoint alone. Even a strong base model can produce unstable output if the surrounding system is weak.

The difficult parts are distributed across the pipeline:
- audio quality affects what the model can hear,
- segmentation affects how much context the decoder sees,
- low-confidence spans need different treatment than easy spans,
- raw transcripts are not always usable without cleanup,
- and changes made during fine-tuning need to be validated in the same runtime conditions in which the model will actually be used.

That is why the repository is organized around a full speech system:
- inference and post-processing at runtime,
- service delivery for downstream consumers,
- and a continuous improvement loop for data and model quality.

In practice, the runtime flow is visible even in the entry-point code:

```python
result = asr.transcribe_file(audio_path, fusion_mode=fusion_mode)
llm_meta = qwen.generate_summary_and_classification(full_text)
```

## Architecture

### Runtime Inference

The runtime path is centered on `src/asr/pipeline.py`, which orchestrates the main transcription workflow. It prepares audio, runs ASR, applies segment-level quality checks, and shapes the final output structure.

At code level, this is anchored by `ASRPipeline` and its `transcribe_file(...)` entry point, which is used from both the CLI and API layers.

Key runtime behavior includes:
- telephony-aware preprocessing before decoding,
- configurable VAD and decoding profiles,
- segment handling through helpers such as `src/asr/segment_utils.py`,
- confidence-aware retry selection and replacement logic in `src/asr/smart_fusion.py`,
- backend abstraction in `src/asr/pipeline_backend.py`,
- and normalized transcript/result models in `src/asr/pipeline_models.py`.

This structure matters because difficult call audio usually fails unevenly. One segment may decode well while another nearby segment fails because of overlap, noise, truncation, or channel changes. The pipeline is designed to recover those local failures instead of treating each file as a single all-or-nothing decode.

Typical failure cases in this stage include:
- noisy telephony audio that hides consonants or damages word boundaries,
- overlapping speakers that make segment boundaries unreliable,
- clipped or too-short spans that decode with weak confidence,
- and unstable segmentation where neighboring spans should really be treated together.

The system handles these cases through a mix of preprocessing, boundary-aware utilities, low-confidence targeting, retry padding, and segment cleanup rather than expecting the base model to solve each issue on its own.

### API / Service Layer

The service layer in `src/api/server.py` exposes the runtime system through a FastAPI application. It separates core transcription from optional downstream text tasks such as summary generation and conversation classification.

The service surface is intentionally simple at the interface level, with routes such as `@app.post("/transcribe")`, while the heavier orchestration stays behind helpers like `_run_asr_pipeline(...)`.

From an engineering perspective, the API layer is more than an HTTP wrapper:
- it coordinates access to GPU-bound inference,
- manages request flow around slower downstream stages,
- normalizes output into structured responses with text, segments, and metadata,
- and supports multiple usage patterns rather than a single endpoint-only workflow.

The repository also keeps a CLI entry point in `src/main.py`, which is useful for local inspection, debugging, and comparing runtime behavior outside the service path.

### Language and Post-Processing

The post-processing layer exists because raw ASR output is rarely the final product users want to consume.

This repository uses multiple forms of cleanup:
- deterministic transcript normalization inside the ASR pipeline,
- lexicon-based correction through `src/hunspell/`,
- optional transformer-based Lithuanian text correction in `src/llm/lithuanian_gec.py`,
- and optional LLM-assisted enrichment via modules such as `src/llm/qwen_classifier.py` and `src/llm/asr_corrector.py`.

The important design choice is that these layers are additive and bounded. The system does not rely on a single opaque post-processing step to “fix everything.” Instead, it combines deterministic cleanup with targeted model-based correction where it is most useful.

That split is visible in code as well:
- deterministic and confidence-driven handling lives in `src/asr/`,
- grammar fixing is exposed via `LithuanianGECFixer.fix_segments(...)`,
- and transcript enrichment is attached through calls such as `generate_summary_and_classification(text)`.

This improves downstream usefulness in two ways:
- transcripts become easier for people to read and review,
- and structured tasks such as summarization or classification start from cleaner text.

It also makes the system boundary clearer:
- inference produces a structured transcript,
- post-processing improves that transcript for human and machine consumption,
- the API exposes those results in stable formats,
- and reviewed outputs can feed back into the model-improvement loop.

### Dataset Building and Fine-Tuning Workflow

The model-development side of the repository lives primarily in `dataset_builder/`. It includes scripts for segmentation, assisted annotation, transcript review, dataset merging, public-data augmentation, and Whisper fine-tuning.

Examples of that workflow include:
- dataset preparation and review helpers such as `dataset_builder/scribe.py` and related scripts,
- augmentation and public speech integration in `dataset_builder/prepare_common_voice.py`,
- fine-tuning orchestration in `dataset_builder/finetune_whisper.py`,
- and additional pipeline utilities under `dataset_builder/eleven_full_context_pipeline/`.

This side of the project shows that model improvement is operationalized, not theoretical. The repository contains the steps needed to go from raw or reviewed training material to a new candidate model and then compare it against earlier versions.

On the implementation side, the training path is anchored by `dataset_builder/finetune_whisper.py`, where the fine-tuning loop is built around Hugging Face training components such as `Seq2SeqTrainer`.

## How the Improvement Loop Connects Back to Inference

The training loop is directly tied to production-style inference concerns.

The feedback cycle looks like this:
- call audio is segmented, reviewed, and cleaned into training-ready examples,
- additional data can be blended in to improve coverage,
- candidate models are fine-tuned and packaged for inference,
- new models are compared against previous versions,
- and the chosen model is then used with tuned decoding, retry, and preprocessing profiles in the runtime system.

In other words, the repository does not treat training and serving as separate worlds. The goal is to improve the actual behavior of the deployed pipeline, not only offline metrics.

## Engineering Trade-Offs

- Retry + fusion versus single-pass decoding: retrying weak spans increases work, but it is usually cheaper and more targeted than rerunning every segment or every file.
- Deterministic cleanup versus model-based correction: deterministic logic handles common, predictable artifacts; model-based correction is reserved for cases where local rules are not enough.
- Quality versus latency: the system keeps core transcription separate from optional enrichment so downstream tasks can be added without making every request pay the highest possible latency cost.

## Operational Considerations

- Inference is GPU-bound, so request coordination matters more than raw HTTP throughput.
- The API separates core transcription from downstream summary/classification work so those stages can be managed independently.
- The implementation favors predictable staged processing over aggressive parallelism where shared compute would otherwise hurt stability or observability.
- Evaluation and comparison scripts are kept close to runtime code so model changes can be checked under similar conditions to the service path.

## Representative Implementation Areas

- `src/main.py`: CLI entry point for running and inspecting the transcription pipeline locally; calls `asr.transcribe_file(...)` and optional text-enrichment steps.
- `src/api/server.py`: FastAPI service layer for transcription, summary, and classification workflows; includes routes such as `@app.post("/transcribe")` and internal orchestration through `_run_asr_pipeline(...)`.
- `src/asr/pipeline.py`: main ASR orchestration, including preprocessing, decoding, retry handling, and output shaping; centered on `ASRPipeline`.
- `src/asr/pipeline_backend.py`: backend abstraction around Faster-Whisper / CTranslate2 inference.
- `src/asr/segment_utils.py`: utility logic for segment merging, retry padding, and gap handling.
- `src/asr/smart_fusion.py`: confidence-aware retry targeting and candidate selection through functions such as `collect_low_confidence_jobs(...)` and `should_replace_segment(...)`.
- `src/llm/qwen_classifier.py`: optional summary and classification layer built on top of transcripts via `generate_summary_and_classification(text)`.
- `src/llm/lithuanian_gec.py`: optional Lithuanian grammar/error correction for transcript text through `fix_segments(...)`.
- `dataset_builder/finetune_whisper.py`: fine-tuning workflow for the domain-adapted Whisper model, built around `Seq2SeqTrainer`.
- `dataset_builder/prepare_common_voice.py`: public-data preparation and augmentation workflow aligned to telephony-style audio.
- `tools/compare_asr_v1_v2.py`: comparison/reporting utility for evaluating successive model versions by running `pipeline.transcribe_file(audio_path)` across model variants.

## Minimal Code Anchors

These examples are intentionally short and illustrative:

```python
jobs = collect_low_confidence_jobs(...)
base_decision = should_replace_segment(...)
```

```python
@app.post("/transcribe")
def transcribe_audio(...):
```

```python
trainer = Seq2SeqTrainer(**trainer_kwargs)
```

They are included only to anchor the documentation to real repository behavior, not to reproduce internal implementation detail.

## What This Demonstrates

This project demonstrates engineering capability across several areas:
- building a real inference pipeline around model behavior instead of stopping at model selection,
- designing service-oriented AI systems with practical runtime constraints in mind,
- adapting speech systems to a language- and domain-specific use case,
- connecting data preparation, fine-tuning, evaluation, and inference into one loop,
- using tests and comparison tooling to support iterative change,
- and making targeted architectural tradeoffs between quality, controllability, and operational simplicity.

The contribution framing in this overview reflects areas I worked on, built, or iterated on within the repository, rather than claiming exclusive ownership of every subsystem end to end.

## Public-Safe Framing

This overview is intentionally generalized for public portfolio use. It avoids internal URLs, private infrastructure details, credentials, exact deployment topology, client-specific business context, and other environment-specific implementation details while preserving the technical shape of the system.
