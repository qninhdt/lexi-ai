# Design — Finetuned Grader (2 task, distilled)

**Status:** design approved, implementation not started
**Date:** 2026-07-29
**Supersedes:** `ml-grader-brainstorm.md` (PARKED — xem §11 các quyết định bị đảo)
**Goal:** học + resume. Không phải giảm cost, không phải giảm latency.

---

## 1. Scope

**Trong scope:** dataset (sinh bằng teacher LLM) → train QLoRA → eval → model card.

**Ngoài scope:** integrate vào `lexi-ai`/`pycil`. Deploy bằng tool sẵn có (vLLM/llama.cpp/Ollama), đổi `llm_base_url` + `llm_api_key` + `llm_model` trong `lexi_ai/config.py`. Không sửa code app trong v1.

**Ngoài scope:** verdict pass/fail. Model chỉ **đo**, app **quyết** (ngưỡng trong `pycil`, tune không cần train lại).

**Teacher:** bất kỳ endpoint OpenAI-compatible, config runtime. Không hardcode.

---

## 2. Hai task

| Task | Việc | Call site |
|---|---|---|
| `use_in_sentence` | Chấm câu learner viết dùng từ target theo 1 nghĩa cụ thể | `lexi_ai/questions/types/use_in_sentence.py` → `grade_rubric` |
| `define` | Chấm định nghĩa learner tự viết cho từ target | chưa có (feature tương lai) |

Ngữ pháp/tự nhiên **không phải task** — là tiêu chí áp cho task 1.

---

## 3. I/O

### Input

```json
{
  "task": "use_in_sentence",
  "target": "bright",
  "sense": { "definition": "full of light", "pos": "adjective" },
  "text": "She's a bright student."
}
```

### Output — `use_in_sentence`

```json
{
  "correction": "She's a bright student.",
  "meaning": 0,
  "feedback": "The sentence is fine, but 'bright' here means 'clever', not 'full of light'."
}
```

### Output — `define`

```json
{
  "meaning": 3,
  "feedback": "Close, but your definition misses the 'persuasive' aspect."
}
```

- `feedback` — **một câu, tiếng Anh**.
- `define` không có `correction`, không có band grammar/naturalness.
- Model sinh **3 field** (task 1) / **2 field** (task 2). Không emit `grammar`/`naturalness` — xem §6.

---

## 4. Format `correction`

Phát lại **cả câu**, markup inline tại chỗ sửa.

```
He [speak>speaks:agr] very [eloquent>eloquently:form].
```

Cú pháp: `[` original `>` replacement `:` tag `]`

| Operation | Cú pháp | Ví dụ |
|---|---|---|
| Thay | `[A>B:tag]` | `[speak>speaks:agr]` |
| Xoá | `[A>:tag]` | `the [the>:art] very` |
| Chèn | `[>B:tag]` | `went [>to the:art] store` |

- Câu sạch → phát verbatim, overhead **0 token**.
- Câu không đọc được → `correction: null` (không cố sinh edit).
- Parse: `\[([^\]>]*)>([^\]:]*):([a-z]+)\]` — một regex, một pass. Ngoài `[...]` là text giữ nguyên.
- Escape: learner gõ `[` → `\[`.
- Chèn `[>B:tag]` đứng riêng giữa hai khoảng trắng, vị trí tường minh trong chuỗi.

**Tại sao dạng này:** phát lại cả câu → frontend render thẳng, không cần khớp span với text gốc. Tag ngắn + tập đóng → message hiển thị nằm ở lookup table phía UI (i18n được), model không sinh chữ giải thích.

**Tham chiếu:** GECToR (Omelianchuk et al., 2020 — Grammarly Research) dùng token tagging, không rewrite, và **không** để model phát ra loại lỗi (ERRANT gán sau). Ta dùng decoder LM nên emit `:tag` trực tiếp gọn hơn dựng ERRANT riêng. Format output nội bộ của sản phẩm Grammarly không công khai — thiết kế dưới đây là của riêng project, không phải bản sao.

---

## 5. Taxonomy — 16 tag

| Nhóm | Trọng số | Tag |
|---|---|---|
| Correctness | 1 | `punc` `sp` `art` `num` `poss` |
| | 2 | `prep` `part` `agr` `tense` `form` `pron` |
| | 3 | `order` |
| Usage | 2 | `coll` `word` |
| | 3 | `unnat` |
| — | 2 | `other` |

`sp` chính tả (gộp CONTR: `dont`) · `agr` hoà hợp chủ-vị · `tense` thời · `form` dạng từ, gộp MORPH/INFL (`eloquent`→`eloquently`) · `art` mạo từ/từ hạn định · `prep` giới từ · `part` tiểu từ phrasal (`look up`/`look on`) · `num` số/đếm được · `poss` sở hữu cách · `pron` đại từ · `order` trật tự từ · `punc` dấu câu · `coll` collocation (`do a decision`) · `word` chọn từ · `unnat` không tự nhiên (gộp register + wordiness) · `other` catch-all

**Không có tag cho sai nghĩa.** Sai nghĩa không sống ở một span, không sửa được bằng replacement → sống ở band `meaning`.

**Tính chất load-bearing:** các tag **dễ nhầm nhau đều cùng trọng số**. `word`↔`coll` (2/2), `prep`↔`part` (2/2) → nhầm không đổi band. Đây là thứ làm việc suy band từ tag chịu được nhãn nhiễu. Mọi thay đổi taxonomy phải giữ tính chất này.

`other` tồn tại để **đo taxonomy thiếu gì**. Không có nó, teacher nhét bừa vào tag gần nhất → nhãn bẩn ẩn.

---

## 6. Band suy từ `correction` (tính ở code, không phải model emit)

```
penalty(nhóm)   = Σ weight(tag ∈ nhóm) / √(số từ)
grammar         = threshold(penalty(Correctness))
naturalness     = threshold(penalty(Usage))
correction null → grammar = 0, bỏ qua công thức
```

Chuẩn hoá `√(số từ)`: 2 lỗi trong 6 từ nặng hơn 2 lỗi trong 30 từ.

**Lợi ích so với để model emit band:**
- **Nhất quán do cấu trúc** — cùng tập lỗi → luôn cùng điểm. Model emit band thì chấm lệch với chính nó.
- Mốc neo tường minh (bảng trọng số) thay vì rubric mơ hồ.
- **Tune được không cần train lại.** Nhãn model emit thì đóng băng vào weights.

**Trọng số + ngưỡng ở trên là giá trị khởi tạo do thiết kế đặt, chưa calibrate.** Calibrate sau khi có dataset: chấm chéo teacher trên vài trăm mẫu → fit ngưỡng.

---

## 7. Rubric `meaning` (model emit)

### `use_in_sentence`

| Band | Tiêu chí |
|---|---|
| 4 | Đúng nghĩa target, không lệch |
| 3 | Đúng nghĩa, lệch nhẹ sắc thái |
| 2 | Mơ hồ — đọc được sang nghĩa khác |
| 1 | Đúng từ, sai sang nghĩa gần |
| 0 | Sai nghĩa hẳn, hoặc không dùng target |

### `define` — chấm thoáng hơn

| Band | Tiêu chí |
|---|---|
| 4 | Bắt đúng nghĩa cốt lõi |
| 3 | Đúng hướng, thiếu một nét |
| 2 | Quá rộng / quá hẹp, vẫn nhận ra nghĩa |
| 1 | Lẫn sang nghĩa khác của cùng từ |
| 0 | Sai hẳn / vòng vo / không nội dung |

`define` chấp nhận lệch 1 band; không phạt diễn đạt.

---

## 8. Sinh dataset — toàn bộ bằng teacher

Input mỗi lần sinh: **1 target + 1 sense** (từ Cambridge DB, `is_extra=0`).

### Hai lượt, bắt buộc tách

```
Lượt 1  SINH:  "viết câu cho {target, sense}, meaning=X, cấy N lỗi loại T"
               → chỉ lấy TEXT
Lượt 2  CHẤM:  chỉ thấy {task, target, sense, text}
               KHÔNG biết band đã yêu cầu, KHÔNG biết nguồn
               → lấy NHÃN (correction, meaning, feedback)
Drop nếu |meaning yêu cầu − meaning nhãn| ≥ 2
```

**Lý do không dùng band yêu cầu làm nhãn:** yêu cầu `meaning=2`, teacher viết ra câu thực tế `meaning=4`. Lấy `2` làm nhãn → dạy model sai, và **không có cách phát hiện sau** (nhãn không lưu dấu vết nó từng là "yêu cầu" chứ không phải "quan sát"). Tỷ lệ drop cũng là số liệu: teacher điều khiển band tốt đến đâu.

Lượt 2 chấm **trộn mọi nguồn trong cùng batch**.

### Grid sinh

Grid chỉ còn **một trục: `meaning` 0–4** (grammar/naturalness suy từ `correction`, không cần quota theo ô).

Trục lỗi điều kiện hoá độc lập: `0 lỗi` · `1 lỗi loại T` · `2–3 lỗi` · `câu không đọc được`.

**Ưu tiên quota vào `meaning` ∈ {1,2,3}** — vùng giữa là chỗ learner thật rơi vào nhiều nhất và khó chấm nhất. Không ép quota thì teacher mặc định viết câu đẹp (`meaning=4`) hoặc sai hẳn (`0`).

Điều kiện hoá thêm khi sinh để text đa dạng: L1 (Vietnamese), CEFR (Cambridge có 16,817 sense gắn nhãn CEFR để chọn độ khó), độ dài, câu đơn/phức, **sai đồng thời nhiều trục**.

### Không có trong v1

- **Input rác** (rỗng, không chứa target, copy nguyên example, sai ngôn ngữ) — rule-detectable → app chặn trước, model không cần học.
- **Prompt injection** — `guarded_messages` (`lexi_ai/llm.py`) đã bọc user turn bằng nonce delimiter.
- Corpus learner thật (W&I+LOCNESS / Lang-8).
- Mọi đường sinh heuristic từ Cambridge (cross-sense, corrupt, collocation/synonym swap).

### Khối lượng

~15–30K row. Task 1 chiếm đa số (task 2 strata mỏng hơn).

---

## 9. Validate sau decode

1. Regex parse toàn bộ `correction`
2. Tag ∈ 16 tập đóng
3. **Strip markup == `text` đầu vào** — bắt buộc. Không có check này, model có thể âm thầm sửa chỗ khác trong câu mà không ai biết (rủi ro riêng của dạng phát-lại-cả-câu)
4. `meaning` ∈ 0–4
5. Không có edit rỗng `[>:tag]`
6. `feedback` một câu, non-empty

Fail → drop row (build data) / retry (inference, `ainvoke_structured` đã có retry).

---

## 10. Split · Eval · Train

### Split

Theo **target word**, không theo row. Một từ sinh nhiều row → chia theo row là leak. Thêm check: hash câu trùng giữa các split.

### Eval

Không có gold (đã defer — §12). Đo được:

| Metric | Ý nghĩa |
|---|---|
| QWK `meaning` student vs teacher (held-out) | fidelity |
| Exact/±1 accuracy per band | phân rã theo vùng |
| `correction` P/R/F1 theo span+tag | so với nhãn teacher |
| Format validity rate | chất lượng constrained decode |
| Phân bố tag; `other` % | taxonomy đủ chưa |
| Ma trận nhầm lẫn tag (chấm 2 lần) | cặp nào lệch **khác trọng số** → cân lại |
| Teacher self-consistency | trần trên của mọi số ở trên |
| Latency p50/p95, VRAM | vận hành |

**Chỉ claim được "distillation fidelity".** Không có số so với literature (đã bỏ BEA-2019/ERRANT). Phải ghi rõ trong model card.

### Train

- Base: Qwen2.5-7B-Instruct hoặc Llama-3.1-8B, **QLoRA**. Colab Pro / Kaggle.
- Constrained/structured decode để ép JSON schema.
- Cân bằng: cap các strata dễ; báo cáo phân bố band trước/sau cân bằng.
- Ablation: full vs `meaning`-only; 7B vs 1.5B (cho câu chuyện latency sau này).

---

## 11. Quyết định đảo so với `ml-grader-brainstorm.md`

| Quyết định cũ | Mới | Lý do |
|---|---|---|
| 3 intent (`correct` / `use_in_context` / `define`) | 2 task | `correct` không phải task — ngữ pháp là tiêu chí, không phải task. Tên đổi thành `use_in_sentence` khớp `type_id` thật trong code |
| BEA-2019 làm nguồn nhãn người | bỏ | Sinh toàn bộ bằng teacher. Hệ quả: mất số so literature |
| ERRANT 11 loại | 16 tag tự định nghĩa | Không cần tương thích ERRANT; thêm `pron`/`part`/`poss`; gộp register+wordiness→`unnat`; thêm `other` |
| XML `<e t="..." c="...">` | `[A>B:tag]` | ~6 token vs ~14 |
| `annotated` + `judgment{verdict,...}` | `correction` + `meaning` + `feedback` | Bỏ verdict (app quyết); grammar/naturalness suy từ `correction` |
| §2: không tag punctuation | giữ `punc` | Task viết câu — thiếu dấu kết là lỗi thật, hiện trên UI |
| §2: loại nhãn rule-base | moot | Sinh toàn bộ bằng teacher |
| `coverage` (define) | `meaning` | Một thang cho cả hai task |
| `naturalness` float 0–1 | band 0–4 | Float là độ chính xác giả; band đo được QWK |
| Base 7–8B | giữ | Mục tiêu học/resume, không phải latency |

---

## 12. Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| **Không có gold** — mọi số neo vào teacher, model kế thừa nguyên bias | cao | Defer có ý thức. Ghi rõ model card. Tạo gold sample sau |
| **Train + test đều teacher-sinh** — không biết có khớp phân phối learner thật | cao | Defer. Đây là mối đe doạ tính hợp lệ lớn nhất |
| Trọng số/ngưỡng band chưa calibrate | trung | Ở code, sửa được không cần train lại |
| `naturalness` không hoàn toàn span-local — câu từng cụm ổn nhưng tổng thể lạ | trung | `unnat` phủ span rộng; là xấp xỉ, không tương đương |
| Teacher không tự nhất quán → distill vô nghĩa | trung | Đo self-consistency trước khi sinh đại trà |
| Model sửa âm thầm chỗ khác trong câu | trung | Validate #3 (strip == input) |
| Không có unseen word (mọi target trong Cambridge 113K) | thấp | Production có `generate_fenced` sinh từ mới. Ghi limitation |
| Output schema ≠ production `Judgment{correct,score,feedback}` | thấp (v1 không deploy) | Khi deploy: PR nhỏ ở `lexi-ai` — prompt + schema + mapping |

---

## 13. Next steps

1. Rubric có mốc neo + prompt teacher (lượt sinh, lượt chấm) cho 2 task.
2. Đo teacher self-consistency trên ~50 mẫu **trước khi** sinh đại trà. Không nhất quán → sửa rubric, không train.
3. Pipeline sinh 2 lượt + validate + drop + cân bằng + split.
4. Sinh pilot ~500 row → histogram từng band + joint distribution → ô nào trống là vùng model sẽ không biết chấm.
5. Calibrate trọng số/ngưỡng band.
6. Sinh full ~15–30K.
7. Train QLoRA + eval harness.
8. Model card + writeup.

---

## Unresolved

- Ngưỡng `threshold(penalty)` → band: chưa có giá trị, cần calibrate (bước 5).
- `define` — có sinh `correction` cho định nghĩa learner viết sai ngữ pháp không? Hiện quyết định: không.
- Bao nhiêu row cho mỗi (target, sense)? Ảnh hưởng độ đa dạng vs chi phí.
- Khi nào tạo gold sample, bao nhiêu mẫu, ai chấm.
- Có ablation 1.5B để lấy số latency cho câu chuyện deploy sau này không.
