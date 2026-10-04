"""Minimal Trainer entry point for the extracted streaming SFT implementation."""
import argparse
import os


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3.5-4B')
    p.add_argument('--model-type', choices=['qwen35', 'qwen3vl'], default='qwen35')
    p.add_argument('--train-jsonl', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--window-chunks', type=int, default=5)
    p.add_argument('--max-length', type=int, default=98304)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--gradient-accumulation', type=int, default=4)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--deepspeed', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = arguments()
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments
    from proactivecoach import model_patch  # Registers the extracted SFT forwards.
    from proactivecoach.attention import register_streaming_attention
    from proactivecoach.collator import DataCollatorForSupervisedDataset
    from proactivecoach.data import StreamingDataset
    from proactivecoach.template import CHAT_TEMPLATE
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    if args.window_chunks < 1 or args.max_length < 1:
        raise ValueError('window-chunks and max-length must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA')
    register_streaming_attention()
    os.environ.setdefault('THINKSTREAM_PAD_BUCKET', '8192')
    os.environ.setdefault('THINKSTREAM_MASK_Q_TILE', '4096')
    processor = AutoProcessor.from_pretrained(args.model)
    processor.chat_template = CHAT_TEMPLATE
    processor.tokenizer.model_max_length = args.max_length
    processor.tokenizer.padding_side = 'right'
    if args.model_type == 'qwen35':
        from transformers import Qwen3_5ForConditionalGeneration as Model
    else:
        from transformers import Qwen3VLForConditionalGeneration as Model
    model = Model.from_pretrained(args.model, dtype=torch.bfloat16,
                                  attn_implementation='streaming_attention')
    model.config.vision_config._attn_implementation = 'flash_attention_2'
    model.config.video_flex_window_size = args.window_chunks
    model.config.use_cache = False
    for p in model.model.visual.parameters():
        p.requires_grad_(False)
    for p in model.model.visual.merger.parameters():
        p.requires_grad_(True)
    for p in model.model.language_model.parameters():
        p.requires_grad_(True)
    model.enable_input_require_grads()
    _apply_liger_kernel_to_instance(model=model, fused_linear_cross_entropy=True)
    data = StreamingDataset(args.train_jsonl, args.video_root, processor,
                            args.model_type, args.max_length)
    collator = DataCollatorForSupervisedDataset(processor.tokenizer,
                                               model.config.text_config.vocab_size)
    training = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=0.1,
        warmup_steps=0.05, lr_scheduler_type='cosine', max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False, label_names=['labels'],
        dataloader_num_workers=args.workers, save_steps=200, save_total_limit=1,
        logging_steps=1, report_to='none', deepspeed=args.deepspeed,
        seed=args.seed, max_steps=args.max_steps,
        ddp_find_unused_parameters=False)

    class SFTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Preserve the original per-microbatch token-mean loss reduction.
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    trainer = SFTTrainer(model=model, args=training, train_dataset=data,
                         data_collator=collator, processing_class=processor)
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
