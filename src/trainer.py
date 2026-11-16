"""SAVRR: Soft Arithmetic-Verified Reference Reweighting on top of OPSD.

Same training loop as OPSDTrainer (on-policy student rollouts, fixed teacher =
base model with LoRA disabled, token-level divergence with point-wise clipping).
The only change is the distillation target:

    p_full = p_T( . | x, r, a, y_<t)        full-reference teacher
    p_ans  = p_T( . | x,    a, y_<t)        answer-only teacher
    p~     = q(r) * p_full + (1 - q(r)) * p_ans          (mixed in probability space)

and the student is trained with   D( p~ || p_theta(. | x, y_<t) )   (beta=0 -> forward KL).

q is supplied per example by the dataset, so the four methods of the study are
one code path:
    opsd         q = 1             (always full reference)
    answer_only  q = 0             (never the rationale)
    binary       q = 1[k == 0]     (binary AVRR)
    savrr        q = q(r)          (soft reliability, Eq. 2)

A teacher context is only evaluated for the rows that need it (q > 0 for the
full-reference teacher, q < 1 for the answer-only teacher), so OPSD and
answer-only cost exactly one teacher forward, like the original trainer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from accelerate.utils import is_peft_model
from transformers import TrainerCallback
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer

from prompts import (  # noqa: E402
    student_user_message,
    teacher_ans_user_message,
    teacher_full_user_message,
)


class SAVRRDataCollator:
    """Builds student, full-reference-teacher and answer-only-teacher prompts.

    Output keys match OPSD's collator (`student_prompts`, `teacher_prompts`, ...)
    so OPSDTrainer.training_step works unchanged; `teacher_prompts` is the
    full-reference context.  Extra keys: `teacher_ans_*` and `q`.

    Expects features with: question, options (dict A-E), correct, answer_text,
    rationale, q.
    """

    def __init__(self, tokenizer, max_length=4096, student_thinking=False, teacher_thinking=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.tokenizer.padding_side = "right"  # same convention as OPSD's collator

    def _chat(self, msg, thinking):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )

    def _encode(self, texts):
        no_pad = self.tokenizer(texts, padding=False, truncation=True, max_length=self.max_length)
        lengths = [len(ids) for ids in no_pad["input_ids"]]
        enc = self.tokenizer(texts, padding="max_length", truncation=True, max_length=max(lengths),
                             return_tensors="pt")
        return enc, lengths

    def __call__(self, features):
        student = [self._chat(student_user_message(f), self.student_thinking) for f in features]
        full = [self._chat(teacher_full_user_message(f), self.teacher_thinking) for f in features]
        ans = [self._chat(teacher_ans_user_message(f), self.teacher_thinking) for f in features]

        s_enc, s_len = self._encode(student)
        f_enc, f_len = self._encode(full)
        a_enc, a_len = self._encode(ans)
        q = torch.tensor([float(f["q"]) for f in features], dtype=torch.float32)
        if (q < 0).any() or (q > 1).any():
            raise ValueError(f"q must be in [0, 1], got {q.tolist()}")

        return {
            "student_prompts": s_enc["input_ids"],
            "student_prompt_attention_mask": s_enc["attention_mask"],
            "student_prompt_length": max(s_len),
            "student_prompt_lengths_per_example": torch.tensor(s_len),
            # full-reference teacher (OPSD's `teacher_*` keys)
            "teacher_prompts": f_enc["input_ids"],
            "teacher_prompt_attention_mask": f_enc["attention_mask"],
            "teacher_prompt_length": max(f_len),
            "teacher_prompt_lengths_per_example": torch.tensor(f_len),
            # answer-only teacher
            "teacher_ans_prompts": a_enc["input_ids"],
            "teacher_ans_prompt_attention_mask": a_enc["attention_mask"],
            "teacher_ans_prompt_length": max(a_len),
            "q": q,
        }


def mix_teacher_log_probs(lp_full, lp_ans, q):
    """log( q * p_full + (1 - q) * p_ans ), row-wise q, computed stably in log space.

    lp_full, lp_ans: [B, T, V] log-probabilities (either may be None if unused)
    q: [B] in [0, 1]
    """
    if lp_ans is None:
        return lp_full
    if lp_full is None:
        return lp_ans
    q = q.to(lp_full.dtype).view(-1, 1, 1)
    # log(0) = -inf is fine inside logsumexp as long as the other term is finite
    return torch.logsumexp(
        torch.stack([lp_full + torch.log(q), lp_ans + torch.log1p(-q)]), dim=0
    )


class VLLMSyncOnResumeCallback(TrainerCallback):
    """After resuming from a checkpoint, push the restored LoRA weights to vLLM once.

    OPSD's GOLDVLLMSyncCallback only syncs every `vllm_sync_frequency` steps, so
    without this the first rollouts after a resume would come from the base weights.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(self.trainer, "use_vllm", False) and state.global_step > 0:
            self.trainer._move_model_to_vllm()
            self.trainer._last_vllm_sync_step = state.global_step


class SAVRRTrainer(OPSDTrainer):
    _tag_names = ["trl", "opsd", "savrr"]
    _name = "SAVRR"

    def __init__(self, *args, student_thinking=False, teacher_thinking=True, pregenerate_full_step=True, **kwargs):
        # [SAVRR-BATCHGEN] pregenerate_full_step: one vLLM call per optimizer step (see get_batch_samples)
        self.pregenerate_full_step = pregenerate_full_step
        if kwargs.get("reason_first"):
            raise ValueError("reason_first is not supported by SAVRRTrainer")
        if kwargs.get("use_thinking_machines_loss"):
            raise ValueError("SAVRR mixes full teacher distributions; use the (default) JSD/KL loss")
        if kwargs.get("data_collator") is None:
            kwargs["data_collator"] = SAVRRDataCollator(
                tokenizer=kwargs["processing_class"],
                max_length=kwargs["args"].max_length,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
            )
        super().__init__(*args, student_thinking=student_thinking, teacher_thinking=teacher_thinking, **kwargs)
        self.add_callback(VLLMSyncOnResumeCallback(self))

    # ------------------------------------------------------------------ [SAVRR-OPTIMIZED]
    # Batched rollout generation.
    # Reduce vLLM calls from 16 to 1 per optimizer step, 
    # Improving generation efficiency and reducing repeated waiting overhead.
    def _can_pregenerate(self):
        return (
            self.pregenerate_full_step
            and self.use_vllm
            and self.vllm_mode == "colocate"
            and not getattr(self, "vllm_enable_sleep_mode", False)
            and getattr(self, "vllm_tensor_parallel_size", 1) == 1
            and not self.reason_first
            and self.processing_class.pad_token_id is not None
        )

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches, device)
        if len(batch_samples) > 1 and self._can_pregenerate():
            self._pregenerate_completions(batch_samples)
        return batch_samples, num_items_in_batch

    def _pregenerate_completions(self, batch_samples):
        pad_id = self.processing_class.pad_token_id
        prompts = [b["student_prompts"] for b in batch_samples]
        width = max(p.shape[1] for p in prompts)
        # right-pad every micro-batch's prompts to a common width; the pad tokens are
        # stripped again when OPSD decodes the prompts for vLLM
        merged = torch.cat([F.pad(p, (0, width - p.shape[1]), value=pad_id) for p in prompts], dim=0)
        self._last_vllm_completion_ids = None
        self._generate_on_policy_outputs_vllm({"student_prompts": merged}, self.generation_config, pad_id)
        completions = self._last_vllm_completion_ids
        if completions is None or len(completions) != merged.shape[0]:
            raise RuntimeError("batched generation returned an unexpected number of completions")
        start = 0
        for b in batch_samples:
            n = b["student_prompts"].shape[0]
            b["pregen_completion_ids"] = [list(c) for c in completions[start:start + n]]
            start += n

    def _set_signature_columns_if_needed(self):
        # keep the raw columns the collator needs
        cols = ["question", "options", "correct", "answer_text", "rationale", "q"]
        if self._signature_columns is None:
            self._signature_columns = cols
        else:
            for c in cols:
                if c not in self._signature_columns:
                    self._signature_columns.append(c)

    # ------------------------------------------------------------------ teacher
    def _teacher_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _teacher_log_probs(self, model, prompts, gen_ids, rows):
        """Teacher log-probs over the generated tokens for the selected rows only."""
        prompt_len = prompts.shape[1]
        ids = torch.cat([prompts[rows], gen_ids[rows]], dim=1)
        mask = torch.ones_like(ids)
        if self.processing_class.pad_token_id is not None:
            mask[ids == self.processing_class.pad_token_id] = 0
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[:, prompt_len - 1: -1, :]
        del out
        lp = F.log_softmax(logits.float() / self.temperature, dim=-1)
        del logits
        return lp

    # ------------------------------------------------------------------ loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        gen_ids = inputs["student_input_ids"][:, student_prompt_len:]
        labels = inputs["labels"][:, student_prompt_len:]
        q = inputs["q"].to(gen_ids.device).float()
        B, T = gen_ids.shape

        # === student ===
        out_s = model(input_ids=inputs["student_input_ids"], attention_mask=inputs["student_attention_mask"])
        student_logits = out_s.logits[:, student_prompt_len - 1: -1, :]
        del out_s

        # === teachers (only the rows that need each context) ===
        rows_full = torch.nonzero(q > 0, as_tuple=True)[0]
        rows_ans = torch.nonzero(q < 1, as_tuple=True)[0]
        with torch.no_grad(), self._teacher_context(model):
            V = student_logits.shape[-1]
            lp_full = lp_ans = None
            if rows_full.numel() > 0:
                lp_full = torch.zeros(B, T, V, device=gen_ids.device, dtype=torch.float32)
                lp_full[rows_full] = self._teacher_log_probs(model, inputs["teacher_prompts"], gen_ids, rows_full)
            if rows_ans.numel() > 0:
                lp_ans = torch.zeros(B, T, V, device=gen_ids.device, dtype=torch.float32)
                lp_ans[rows_ans] = self._teacher_log_probs(model, inputs["teacher_ans_prompts"], gen_ids, rows_ans)
            # rows that only have one context: q is exactly 0 or 1, so the mixture
            # weight of the missing (zero-filled) context is exactly 0 -> no effect.
            if lp_full is not None and lp_ans is not None:
                teacher_lp = mix_teacher_log_probs(lp_full, lp_ans, q)
            else:
                teacher_lp = lp_full if lp_full is not None else lp_ans
            del lp_full, lp_ans
            empty_cache()

        # === loss ===
        # generalized_jsd_loss applies log_softmax(logits / temperature).  Pass the
        # already temperature-scaled student logits and the teacher *log-probs*
        # (log_softmax of normalised log-probs is the identity) with temperature=1.
        loss = self.generalized_jsd_loss(
            student_logits=student_logits.float() / self.temperature,
            teacher_logits=teacher_lp,
            labels=labels,
            beta=self.beta,
            temperature=1.0,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        del student_logits, teacher_lp
        empty_cache()

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["savrr/q_mean"].append(q.mean().item())
        self._metrics[mode]["savrr/frac_full_only"].append((q == 1).float().mean().item())
        self._metrics[mode]["savrr/frac_ans_only"].append((q == 0).float().mean().item())
        self._metrics[mode]["savrr/frac_mixed"].append(((q > 0) & (q < 1)).float().mean().item())

        if return_outputs:
            class _Out:
                pass
            o = _Out()
            o.loss = loss
            return loss, o
        return loss
