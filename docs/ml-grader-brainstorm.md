# Brainstorm — Multi-task LLM Grader (SUPERSEDED)

**Status:** superseded by [`grader-finetune-design.md`](./grader-finetune-design.md) (2026-07-29)
**Date:** 2026-07-24

> Doc này giữ lại làm hồ sơ lập luận. Thiết kế thi hành nằm ở
> `grader-finetune-design.md` §11 (bảng các quyết định đã đảo). Điểm đảo lớn:
> 3 intent → 2 task · bỏ BEA-2019/ERRANT · XML → `[A>B:tag]` ·
> bỏ `verdict` · grammar/naturalness suy từ `correction` thay vì model emit.
**Context:** dùng lexi-ai làm project cho CV AI engineer — thực hành fine-tuning (LLM + DL-for-NLP) đồng thời giảm phụ thuộc gọi LLM. Doc này chốt thiết kế task LLM (grader) để tiếp tục sau. Task encoder nhãn-tay (CEFR classification / WSD) brainstorm riêng.

---

## 1. Audit — LLM call sites & data assets

LLM call đi qua seam `StructuredLLM.parse` (`lexi_ai/llm.py`) — dễ swap. Call sites (từ `prompts/`): `senses_generation` (synth entry đầy đủ), `wsd`, `example_augment`, `contextual_mcq`, `rubric_scoring`, themed_*, translate. Sense embeddings đã local (MiniLM).

Hai mỏ dữ liệu:
- **Cambridge DB** (`./data`, 142MB, read-only): 202,607 senses, 529,266 examples, 113K words. **16,817 senses có nhãn CEFR** (A1:855 / A2:1,710 / B1:3,265 / B2:4,698 / C1:2,497 / C2:3,792); 14,708 có domain; 1,853 topics + 122,654 word→topic; 30,974 sense_synonyms.
- **Generation cache** (`examples-lexi.db`): mỗi entry cache = anchor→JSON, nhưng **chỉ lưu dạng phân rã quan hệ, KHÔNG lưu raw teacher JSON** (`persist_result`).

## 2. Quyết định bị loại (giữ lại lý do — đừng re-litigate)

- **Distill toàn bộ `senses_generation` → student nhỏ: LOẠI.** Project = lazy + cache vĩnh viễn + vocab hữu hạn (~113K, hữu ích ~20–30K). Tổng chi phí ≈ (số từ unique) × 1 lần generate; cache làm lần tra lại = 0 → **đã tối ưu**. Student chỉ tiết kiệm phần từ chưa gen, mà để train student phải gen phần lớn số đó bằng teacher trước → bị dominated. Data cũng là synthetic teacher. Vô nghĩa cho vocab bounded.
- **Grading task có nhãn dựng-bằng-rule (perturbation/synonym-swap): LOẠI.** Rule-constructible ⟹ rule-gradable ⟹ LLM thừa. Task đáng dùng LLM ⟺ output mở + chất lượng holistic (không rule hoá được) ⟹ nhãn buộc từ teacher/người.
- **CEFR facet trong grader: bỏ tạm.** (CEFR-của-từ tách sang encoder task riêng.)
- **Lỗi hình thức (capitalization/whitespace/orthography/punctuation): không tag.**

## 3. Vì sao LLM chính đáng ở grading (khác generator)

Input grading = câu/bài learner viết ra = **vô hạn, không cache được** → chi phí tăng theo **usage** chứ không theo vocab. Local grader distilled amortize thật. Đây là điểm khác bản chất so với generation (bounded + cacheable).

## 4. Thiết kế chốt — một model, một output shape

Multi-task instruction tuning (paradigm T5/FLAN). **Model là một**, "chia task" chỉ ở tầng dữ liệu (mỗi mẫu 1 `intent`). Base **QLoRA 7–8B** (Qwen2.5-7B / Llama-3.1-8B). Chạy được trên Kaggle/Colab Pro.

**3 intent:** `correct` (chỉ sửa lỗi — dùng data GEC gold), `use_in_context` (chấm câu learner dùng từ target), `define` (chấm định nghĩa learner tự viết).

**Bộ loại lỗi (11, coarsen từ ERRANT category chính):**
`spelling · article · preposition · agreement · tense · verb-form · noun-number · word-form · word-choice · word-order · other`
(collocation/register → feedback text, không tag. Không tag lỗi hình thức.)

Operation qua cấu trúc tag (không cần loại riêng):
```
thay:  <e t="preposition" c="on">in</e>
thiếu: <e t="article" c="the"/>
thừa:  <e t="article" c="">the</e>
```

### Input
```json
{ "intent": "correct | use_in_context | define",
  "target": "eloquent",
  "sense": {"definition":"fluent and persuasive","pos":"adj"},
  "text": "He speak very eloquent." }
```

### Output — `annotated` luôn có (facet lỗi), `judgment` theo intent
```json
// use_in_context
{ "annotated":"He <e t=\"agreement\" c=\"speaks\">speak</e> very <e t=\"word-form\" c=\"eloquently\">eloquent</e>.",
  "judgment":{"intent":"use_in_context","verdict":true,"sense_match":true,"naturalness":0.6,
              "feedback":"Đúng nghĩa nhưng sai hoà hợp chủ-vị + cần dạng trạng từ."} }

// define
{ "annotated":"It means <e t=\"word-choice\" c=\"speaking\">to talk</e> a lot .",
  "judgment":{"intent":"define","verdict":false,"coverage":0.4,
              "feedback":"Thiếu nét 'thuyết phục/trôi chảy'."} }

// correct
{ "annotated":"I <e t=\"tense\" c=\"went\">go</e> there yesterday .",
  "judgment":{"intent":"correct"} }
```
Một parser, một grammar decode. Validate JSON + XML well-formed + tag ∈ 11 loại sau decode.

## 5. Dataset — HYBRID (không chọn cực đoan)

| Nguồn | Sub-task | Cách |
|-------|----------|------|
| **BEA-2019 (W&I+LOCNESS)** +opt FCE/NUCLE | `correct` (facet lỗi) | **Dataset người-gán sẵn** — KHÔNG distill (teacher cũng dở GEC). Parse M2 → map ERRANT ~55 → 11 loại thô, lọc bỏ ORTH/PUNCT/caps → convert sang chuỗi `<e>` inline |
| Teacher distill (Cambridge target words) | `use_in_context` | Teacher sinh câu learner đa chất lượng (đúng/sai sense/sai collocation/lỗi ngữ pháp; +opt lọc câu learner thật Lang-8/BEA chứa target), rồi teacher chấm nguyên khối `{annotated, judgment}` |
| Teacher distill | `define` | Teacher mô phỏng định nghĩa learner nhiều mức → chấm `{annotated, judgment(coverage,verdict,feedback)}` vs sense tham chiếu |

- **Co-train facet lỗi:** mỗi mẫu use_in_context/define teacher điền luôn `annotated` → facet lỗi học từ cả GEC gold (typing chặt) + câu in-domain quanh target. Miễn phí do 1 shape.
- **Validate + cân bằng:** bỏ output teacher hỏng schema; cap GEC gold để không đè 2 task distill; split theo word/document tránh leakage.
- **Khối lượng gợi ý:** GEC ~34K (W&I sẵn) · use_in_context ~5–15K · define ~3–8K → ~50K (10–20K đã đủ cho QLoRA).

## 6. Eval (3 tier)

- **Facet lỗi:** map `<e>` → M2, chấm **ERRANT trên BEA test** → số so được literature.
- **use_in_context/define:** **tự gán test set nhỏ ~200–500 (Tier 2)** → correctness trên đúng task; +opt public set chỗ khớp (SemEval LexSub cho lexical, ASAP/W&I cho scoring).
- Train 100% teacher-distill cho 2 task judgment; báo số trên nhãn người ở eval.
- Nếu bỏ hết human data → chỉ claim được "distillation fidelity" (yếu hơn cho CV).

## 7. Rủi ro / câu hỏi mở

- error_analysis là task khó nhất — bắt buộc coarsen + validate XML.
- Negative transfer nếu data lệch → cân bằng + eval RIÊNG từng task.
- Grading safety-sensitive → giữ đường abstain theo confidence + spot-audit, chưa bỏ hẳn teacher fallback ca biên.
- Teacher sinh cả input (câu learner) lẫn label → hơi circular; eval trên human test set để bắt bias.
- CEFR trên input ngắn nhiễu (đã bỏ tạm).
- **Mở:** có thêm intent nào (paraphrase? register?) không — hiện giữ 3.

## 8. Companion task (không-LLM, nhãn-tay)

Vế "DL cho NLP" = **1 encoder task nhãn-tay**: WSD cross-encoder (SemCor/WordNet) **hoặc** CEFR classification (16,817 sense Cambridge). → đang brainstorm riêng.

## 9. Next steps khi resume

1. Chốt base model + môi trường (Kaggle/Colab Pro).
2. Viết bộ chuyển M2 → `<e>` + bảng map ERRANT→11 loại.
3. Prompt teacher cho use_in_context/define (sinh input + chấm).
4. Pipeline validate/balance/split.
5. Harness eval (ERRANT + hand-labeled Tier-2).
6. → `/ck:plan` khi bắt tay triển khai.
