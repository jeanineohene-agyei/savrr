"""Prompt templates shared by SAVRR training and AQuA evaluation."""

BOX_INSTRUCTION = (
    "Please reason step by step, and put the letter of the correct option (A, B, C, D, or E) "
    "within \\boxed{}."
)

FULL_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)
ANS_TRANSITION = (
    "\n\nNow, using your own words and independent reasoning, derive this final answer "
    "to the problem above. Think step by step, explore different approaches, and don't be "
    "afraid to backtrack or reconsider if something doesn't work out:\n"
)


def format_problem(question: str, options: dict) -> str:
    opts = "\n".join(f"{k}) {options[k]}" for k in sorted(options))
    return f"Problem: {question}\n\nOptions:\n{opts}"


def student_user_message(ex) -> str:
    return f"{format_problem(ex['question'], ex['options'])}\n\n{BOX_INSTRUCTION}"


def teacher_full_user_message(ex) -> str:
    return (
        f"{format_problem(ex['question'], ex['options'])}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{ex['rationale']}\n=== Reference Solution End ===\n"
        f"The correct answer is ({ex['correct']}) {ex['answer_text']}."
        f"{FULL_TRANSITION}\n{BOX_INSTRUCTION}"
    )


def teacher_ans_user_message(ex) -> str:
    return (
        f"{format_problem(ex['question'], ex['options'])}\n\n"
        f"The correct answer to this problem is ({ex['correct']}) {ex['answer_text']}."
        f"{ANS_TRANSITION}\n{BOX_INSTRUCTION}"
    )
