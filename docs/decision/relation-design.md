
# Sense relation resolution — supplied design

Original annotated specification retained below; the executable JSON-e template
lives in `lexi_ai/relations/prompts/resolve_sense_relations.json`.

// # 1. Mục tiêu

// Input là một semantic relation chưa resolve:

// ```text
// source sense
//     │
//     ├── relation_type
//     │
//     ├── target_word
//     │
//     └── target_gloss
//               │
//               ▼
//        target sense inventory
//               │
//               ▼
//         sense_id | null
// ```

// Task chính xác là:

// > Chọn sense của `target_word` vừa phù hợp với intended target meaning được mô tả bởi `target_gloss`, vừa thực sự tạo thành `relation_type` với source sense.

// Về mặt NLP, đây gần với **sense linking / word-sense alignment** hơn WSD cổ điển. Classic WSD chọn sense cho một word occurrence trong context; gloss-based systems như GlossBERT cũng formulate WSD dưới dạng context–gloss matching. citeturn807180search2

// ---

// # 2. Final template

// Backend trước tiên lấy **toàn bộ candidate senses có POS tương thích** rồi build template này.

// ```json
// {
//   "state": {
//     "source": {
//       "word": "${source_word}",
//       "definition": "${source_definition}"
//     },

//     "relation": {
//       "type": "${relation_type}",
//       "rule": {
//         "$switch": {
//           "relation_type == 'synonym'":
//             "The target sense must express essentially the same lexicalized concept as the source sense.",

//           "relation_type == 'antonym'":
//             "The target sense must express an opposing meaning to the source sense on the relevant semantic dimension.",

//           "relation_type == 'hypernym'":
//             "The source sense must denote a kind or type of the target sense.",

//           "relation_type == 'hyponym'":
//             "The target sense must denote a kind or type of the source sense.",

//           "relation_type == 'meronym'":
//             "The target sense must denote a part, member, or substance of what the source sense denotes.",

//           "relation_type == 'holonym'":
//             "What the source sense denotes must be a part, member, or substance of what the target sense denotes."
//         }
//       }
//     },

//     "target": {
//       "word": "${target_word}",
//       "gloss": "${target_gloss}"
//     }
//   },

//   "questions": {
//     "matched_sense": {
//       "type": "choice",

//       "instructions": "Which candidate sense of `target.word` is the intended endpoint of this semantic relation? A candidate must both be compatible with `target.gloss` and satisfy `relation.rule` with the source sense. Treat `target.gloss` as evidence of the intended target meaning, not as authoritative ground truth. Prefer semantic compatibility over wording overlap. Choose `no_candidate` if no candidate satisfies both requirements.",

//       "criteria": {
//         "$reduce": {
//           "$eval": "candidates"
//         },

//         "initial": {
//           "no_candidate": "No candidate sense both represents the intended target meaning and forms the required semantic relation with the source sense."
//         },

//         "each(acc, sense)": {
//           "$merge": [
//             {
//               "$eval": "acc"
//             },
//             {
//               "candidate_${sense.index}": "${target_word} - ${sense.pos} - ${sense.definition}"
//             }
//           ]
//         }
//       }
//     }
//   }
// }
// ```

// Model chỉ thấy:

// ```text
// candidate_1
// candidate_2
// ...
// no_candidate
// ```

// Không thấy database `sense_id`.

// Backend giữ mapping:

// ```text
// candidate_1 → actual sense_id
// candidate_2 → actual sense_id
// ...
// ```

// Final public result của resolver chỉ là:

// ```text
// sense_id: UUID | null
// ```

// ---

// # 3. Candidate loading

// Current implementation đã đúng khi **lọc POS trong SQL trước candidate limit**. fileciteturn0file0L78-L85

// Design mới:

// ```text
// target_word
//     ↓
// target word generation_state == done
//     ↓
// all senses compatible with source POS
//     ↓
// 0 candidates?
//     ├─ yes → null / unresolvable
//     └─ no  → decision model
// ```

// ## Không còn `LIMIT 12`

// Current system lấy same-POS candidates rồi giới hạn 12. fileciteturn0file0L128-L141

// Bỏ giới hạn semantic này.

// Phải đưa:

// ```text
// ALL compatible candidate senses
// ```

// vào choice.

// Nếu correct sense nằm ngoài arbitrary top 12 thì không model nào có thể cứu được.

// Nếu sau này có một word pathological với số sense vượt model limit, đó phải là một explicit exceptional path; **không silently truncate candidate inventory rồi coi kết quả là resolved**.

// ---

// # 4. Vì sao giữ POS filter?

// WordNet tổ chức một sense như một meaning cụ thể của word trong một POS; hypernym/hyponym, meronym/holonym và các semantic pointers cũng được định nghĩa ở sense/synset level. citeturn807180search0turn807180search1

// Với relation set hiện tại:

// ```text
// synonym
// antonym
// hypernym
// hyponym
// meronym
// holonym
// ```

// same-POS filtering là hợp lý.

// Nếu sau này thêm relation kiểu:

// ```text
// derivationally_related
// pertains_to
// participle_of
// ```

// thì không còn dùng global same-POS rule; chuyển sang:

// ```text
// relation_type
// → allowed source/target POS combinations
// ```

// WordNet cũng tách riêng các cross-POS pointers này. citeturn807180search1

// ---

// # 5. Hai nguồn evidence

// Resolver phải dùng đồng thời:

// ```text
// A. target_gloss
// B. source sense + semantic relation
// ```

// Không được biến task thành:

// ```text
// candidate nào giống target_gloss nhất?
// ```

// ## `target_gloss`

// Nó trả lời:

// > Upstream generator có ý định nói tới target meaning nào?

// Nhưng vì gloss do model sinh nên nó có thể:

// - thiếu detail
// - paraphrase không đẹp
// - hơi inaccurate
// - chứa lexical overlap gây nhiễu

// Do đó nó chỉ là **evidence**.

// ## `source + relation`

// Đây là semantic constraint thứ hai.

// Ví dụ:

// ```text
// source:
// dog = a domesticated canine

// relation:
// hypernym

// target_word:
// animal
// ```

// Rule phải được hiểu theo direction:

// ```text
// dog IS-A animal
// ```

// tức:

// ```text
// source is a kind of target
// ```

// WordNet chính thức định nghĩa hypernym theo đúng hướng `X is a kind of Y`. citeturn807180search0turn807180search4

// ---

// # 6. Relation semantics

// Các rule phải nằm explicit trong template thay vì chỉ gửi:

// ```json
// {
//   "relation_type": "hypernym"
// }
// ```

// ### synonym

// ```text
// source ≈ target
// ```

// Target phải lexicalize essentially the same concept.

// ### antonym

// ```text
// source ↔ semantic opposite of target
// ```

// Phải opposition trên relevant semantic dimension, không chỉ “khác nghĩa”.

// ### hypernym

// ```text
// SOURCE is a kind/type of TARGET
// ```

// Ví dụ:

// ```text
// dog → animal
// ```

// ### hyponym

// ```text
// TARGET is a kind/type of SOURCE
// ```

// Ví dụ:

// ```text
// animal → dog
// ```

// ### meronym

// ```text
// TARGET is a part/member/substance of SOURCE
// ```

// Ví dụ:

// ```text
// car → wheel
// ```

// ### holonym

// ```text
// SOURCE is a part/member/substance of TARGET
// ```

// Ví dụ:

// ```text
// wheel → car
// ```

// WordNet cũng phân biệt member, substance và part variants của meronym/holonym. citeturn807180search1

// ---

// # 7. `no_candidate`

// `no_candidate` là first-class result, không phải model failure.

// Nó có nghĩa:

// > Không candidate nào đồng thời phù hợp intended target meaning **và** thỏa semantic relation với source sense.

// Ví dụ:

// ```text
// source:
// dog = domesticated canine

// relation:
// synonym

// target_word:
// table

// target_gloss:
// a piece of furniture
// ```

// Candidate:

// ```text
// table = a piece of furniture
// ```

// match gloss hoàn hảo.

// Nhưng:

// ```text
// dog ≠ synonym(table)
// ```

// nên:

// ```text
// no_candidate
// ```

// Đây là lý do relation constraint phải tham gia decision.

// ---

// # 8. Decision flow

// ```text
// Pending Relation
//       │
//       ▼
// Target word generated + has senses?
//       │
//       ├── no
//       │    └─ remain pending / not eligible
//       │
//       ▼
// Load ALL compatible-POS senses
//       │
//       ├── zero
//       │    └─ unresolvable
//       │
//       ▼
// Build final state
//       │
//       ├── source word
//       ├── source definition
//       ├── relation type
//       ├── explicit relation rule
//       ├── target word
//       └── target gloss
//       │
//       ▼
// Jev Choice
//       │
//       ├── no_candidate
//       │       └─ selected = null
//       │
//       └── candidate_N
//               │
//               ▼
//           confidence
//           ┌────┴─────┐
//         high         low
//          │            │
//          │            ▼
//          │       LLM fallback
//          │            │
//          └─────┬──────┘
//                ▼
//         sense_id | null
//                │
//                ▼
//       transactional revalidation
//                │
//          ┌─────┴─────┐
//        stale         valid
//          │             │
//         noop          commit
// ```

// ---

// # 9. Jev → LLM fallback

// Current architecture đã có:

// ```text
// Jev
// → confidence threshold
// → low confidence
// → OpenAI Structured Outputs fallback
// ```

// fileciteturn0file0L180-L187

// Giữ.

// Cả hai model phải nhận cùng semantic contract:

// ```text
// source
// relation + explicit rule
// target word
// target gloss
// all candidates
// no_candidate
// ```

// Fallback không được có semantics khác với Jev.

// ## Threshold

// `LEXI_DECISION_THRESHOLD` phải calibrate bằng validation dataset thay vì chọn tùy ý.

// Dataset:

// ```text
// source sense
// relation
// target word
// target gloss
// candidate senses
// gold sense_id | null
// ```

// Theo từng threshold đo:

// ```text
// resolved precision
// resolved recall
// fallback rate
// false-resolution rate
// ```

// Ở đây nên ưu tiên:

// ```text
// precision > recall
// ```

// vì:

// ```text
// false null
// ```

// chỉ khiến edge chưa resolve,

// trong khi:

// ```text
// false positive sense
// ```

// ghi semantic relation sai vào lexicon.

// ---

// # 10. Database state

// Giữ ba trạng thái hiện tại:

// ```text
// pending
// resolved
// unresolvable
// ```

// Current semantics đã hợp lý:

// - `pending`: chưa xử lý hoặc result bị stale
// - `resolved`: có `to_sense_id` hợp lệ
// - `unresolvable`: đã thử nhưng không có target sense

// fileciteturn0file0L104-L112

// Không cần thêm:

// ```text
// low_confidence
// ambiguous
// model_rejected
// ...
// ```

// những thứ đó là execution detail, không phải persistent semantic states.

// Transport/model error cũng không biến edge thành `unresolvable`.

// ---

// # 11. Commit protocol

// Phần hiện tại giữ nguyên.

// Model inference xảy ra ngoài transaction.

// Sau khi có decision:

// ```text
// SELECT relation FOR UPDATE
// ```

// sau đó revalidate:

// ```text
// source_word
// source_definition
// relation_type
// source_pos
// target_word
// target_gloss
// candidate inventory + candidate definition hashes
// ```

// Nếu khác snapshot đã đưa cho model:

// ```text
// noop
// ```

// không commit stale decision.

// Nếu giống:

// ```text
// candidate
// → set to_sense_id
// → target_hash
// → resolve_attempted_at

// no_candidate
// → to_sense_id = null
// → resolve_attempted_at
// ```

// Current implementation đã dùng `SELECT FOR UPDATE` + snapshot revalidation trước update, nên nguyên tắc này giữ nguyên. fileciteturn0file0L189-L200

// ---

// # 12. Invalidation

// Giữ database-trigger invalidation.

// Nếu source hoặc target semantic data thay đổi:

// ```text
// resolved/unresolvable
// → pending
// ```

// để relation được evaluate lại.

// Current triggers đã invalidate relations khi neutral definitions hoặc relevant sense metadata thay đổi. fileciteturn0file0L203-L220

// Điều này đặc biệt quan trọng sau khi design mới phụ thuộc vào cả:

// ```text
// source_definition
// target candidate definitions
// ```

// ---

// # 13. Concurrency

// Giữ:

// ```text
// resolve_relations(batch_size=N)
// → load batch
// → asyncio.gather(per relation)
// ```

// Một edge lỗi:

// ```text
// error(edge)
// ```

// không rollback/cancel các edge khác.

// Current design đã có per-link error isolation. fileciteturn0file0L71-L77

// Đây là đúng abstraction vì mỗi relation là một independent reconciliation unit.

// ---

// # 14. Final output contract

// Decision layer:

// ```text
// candidate_N | no_candidate
// ```

// Resolver layer convert thành:

// ```text
// sense_id | null
// ```

// Persistent relation:

// ```text
// to_sense_id = sense_id | null
// ```

// Public API có thể trả:

// ```json
// {
//   "edge_id": "...",
//   "state": "resolved",
//   "sense_id": "..."
// }
// ```

// hoặc:

// ```json
// {
//   "edge_id": "...",
//   "state": "unresolvable",
//   "sense_id": null
// }
// ```

// Không cần expose:

// ```text
// candidate index
// Jev probability distribution
// fallback model choice
// DB candidate IDs
// ```

// trừ telemetry/debugging nội bộ.

// ---

// # 15. Final architecture

// ```text
// generate_word
//     │
//     ▼
// creates half-edge
// (from_sense, relation, target_word, target_gloss)
//     │
//     │ explicit resolve_relations()
//     ▼
// pending_relations()
//     │
//     ├─ target not ready → skip
//     │
//     ▼
// same-POS candidate lookup
//     │
//     ├─ 0 candidates → unresolvable
//     │
//     ▼
// ALL candidate senses
//     │
//     ▼
// Sense Relation Resolution Template
//     │
//     ├─ source meaning
//     ├─ explicit relation semantics
//     ├─ target gloss
//     └─ candidate definitions
//     │
//     ▼
// Jev
//     │
//     ├─ confident → result
//     │
//     └─ uncertain → LLM fallback
//     │
//     ▼
// candidate | null
//     │
//     ▼
// candidate → sense_id
//     │
//     ▼
// FOR UPDATE + snapshot revalidation
//     │
//     ├─ stale → noop / retry later
//     │
//     └─ valid
//           │
//           ▼
//          commit
//           │
//           ▼
// resolved | unresolvable
// ```

// # 16. Những thay đổi cuối cùng so với design cũ

// | Phần | Final |
// |---|---|
// | half-edge model | giữ |
// | explicit on-demand resolution | giữ |
// | same-POS SQL filter | giữ |
// | `LIMIT 12` | **bỏ** |
// | candidate inventory | **all compatible senses** |
// | source definition | giữ |
// | target gloss | giữ nhưng coi là **evidence** |
// | relation type | giữ |
// | relation semantics | **thêm explicit rule** |
// | candidate scoring | **gloss compatibility + relation validity** |
// | `no_candidate` | giữ, semantics chặt hơn |
// | expose DB ID cho model | không |
// | Jev Choice | giữ |
// | confidence fallback | giữ |
// | arbitrary threshold | **thay bằng calibrated threshold** |
// | output NLP | `sense_id | null` |
// | lock + snapshot revalidation | giữ |
// | invalidation triggers | giữ |
// | concurrency/error isolation | giữ |

// **Mental model cuối cùng:**

// ```text
// Không phải:

// target_gloss
// → candidate có gloss giống nhất


// Mà là:

//                   target_gloss
//                        │
//                        ▼
// source sense ── relation rule ──► candidate sense
//                        │
//                        ▼
//           both constraints satisfied?
//                  │          │
//                 yes         no
//                  │          │
//              sense_id      null
// ```
