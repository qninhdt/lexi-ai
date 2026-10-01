# Definition and usage grading — supplied design

Original annotated specification retained below; executable JSON-e templates live
in `lexi_ai/questions/prompts/decision/`.

{
  "state": {
    "word": "${word.lemma}",
    "answer": {
      "$eval": "answer"
    }
  },

  "questions": {
    "defined_meaning": {
      "type": "choice",
      "instructions": "Which listed sense of `word` does `answer` most clearly attempt to define? Select the best-supported intended sense even if the definition is incomplete, vague, or contains some semantic error. Choose `no_candidate` only when there is not enough semantic evidence to identify any listed sense.",

      "criteria": {
        "$reduce": {
          "$eval": "word.senses"
        },

        "initial": {
          "no_candidate": "No listed sense is reasonably identifiable from the answer."
        },

        "each(acc, sense)": {
          "$merge": [
            {
              "$eval": "acc"
            },
            {
              "sense_${sense.id}": "${word.lemma} - ${sense.pos} - ${sense.definition}"
            }
          ]
        }
      }
    }
  }
}

// Pipeline mới cho nhóm **`word_to_*`** chốt như này:

// ## 1. `word_to_definition`

// Input:

// ```text
// word
// answer
// all senses của word
// ```

// ### Query 1 — resolve sense user đang định nghĩa

// ```text
// word + answer + candidate senses
//         ↓
// defined_meaning
//         ↓
// sense_id | no_candidate
// ```

// Nếu:

// ```text
// defined_meaning = no_candidate
// ```

// thì dừng:

// ```text
// sense_id = null
// accuracy = null
// coverage = null
// ```

// Nếu resolve được sense:

// ```text
// selected sense
//     ↓
// Query 2
// ```

// ### Query 2 — chấm semantic knowledge

// Input chỉ còn:

// ```text
// word
// selected meaning
// answer
// ```

// Output:

// ```text
// accuracy
// → accurate | mixed | inaccurate

// coverage
// → sufficient | partial | minimal
// ```

// Final result:

// ```json
// {
//   "sense_id": "...",
//   "accuracy": "accurate",
//   "coverage": "partial"
// }
// ```

// Mental model:

// ```text
// word_to_definition
//         │
//         ▼
// resolve intended sense
//         │
//         ├── none → stop
//         │
//         ▼
// grade selected sense
//         │
//         ├── accuracy
//         └── coverage
// ```

// ---

// ## 2. `word_to_usage`

// Input:

// ```text
// word
// required meaning
// answer
// ```

// Ở đây target sense/meaning **đã biết trước**, nên Query 1 không cần sense resolution.

// ### Query 1 — gate `used`

// ```text
// word + answer
//     ↓
// used?
// ```

// Nếu:

// ```text
// used = false
// ```

// thì dừng:

// ```text
// meaning = null
// form = null
// construction = null
// collocation = null
// appropriacy = null
// ```

// Nếu:

// ```text
// used = true
// ```

// thì chạy Query 2.

// ### Query 2 — grade usage

// Input:

// ```text
// word
// required meaning
// answer
// ```

// Output:

// ```text
// meaning
// → correct | approximate | wrong

// form
// → correct | spelling_error | form_error

// construction
// → true | false

// collocation
// → natural | acceptable | unnatural

// appropriacy
// → appropriate | marked | inappropriate
// ```

// Final result:

// ```json
// {
//   "used": true,
//   "meaning": "correct",
//   "form": "correct",
//   "construction": true,
//   "collocation": "acceptable",
//   "appropriacy": "appropriate"
// }
// ```

// Mental model:

// ```text
// word_to_usage
//       │
//       ▼
// was target used?
//       │
//       ├── no → stop
//       │
//       ▼
// grade productive usage
//       │
//       ├── meaning
//       ├── form
//       ├── construction
//       ├── collocation
//       └── appropriacy
// ```

// ---

// ## 3. Điểm chung của cả hai pipeline

// Architecture chung là:

// ```text
// Query 1 = gate / route
// Query 2 = diagnostic grading
// ```

// Nhưng **không dùng chung criteria**:

// ```text
// word_to_definition
// → semantic representation
// → accuracy + coverage
// ```

// ```text
// word_to_usage
// → productive lexical use
// → meaning + form + construction + collocation + appropriacy
// ```

// Chỉ nên share orchestration/code infrastructure:

// ```text
// gate
// → early stop
// → second Jev query
// → compose result
// ```

// không share rubric.

// ### Tổng thể

// ```text
//                     word_to_*
//                         │
//           ┌─────────────┴─────────────┐
//           │                           │
//  word_to_definition             word_to_usage
//           │                           │
//  Query 1: sense resolution      Query 1: used gate
//           │                           │
//       no candidate                    false
//           │                           │
//          stop                        stop
//           │                           │
//      selected sense                  true
//           │                           │
//         Query 2                     Query 2
//           │                           │
//  accuracy + coverage        meaning + form +
//                             construction +
//                             collocation +
//                             appropriacy
// ```

// Đây là pipeline mới em sẽ chốt.
