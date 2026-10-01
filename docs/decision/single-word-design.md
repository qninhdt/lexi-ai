# Single-word grading — supplied design

Original annotated specification retained below; executable JSON-e templates live
in `lexi_ai/questions/prompts/decision/`.

{
  "state": {
    "question": {
      "$eval": "question"
    },
    "answer": {
      "$eval": "answer"
    }
  },

  "questions": {
    "task_fit": {
      "$switch": {
        "question_type == 'definition_to_word'": {
          "type": "noul",
          "instructions": "Does the lexical expression clearly intended by `answer` accurately satisfy the definition in `question`? Accept a different lexical expression if it fully expresses the defined meaning. Ignore only clear spelling errors; do not repair other lexical or grammatical differences.",
          "criteria": {
            "true": "The intended lexical expression accurately satisfies the definition.",
            "false": "The intended lexical expression does not accurately satisfy the definition."
          }
        },

        "question_type == 'context_to_word'": {
          "type": "noul",
          "instructions": "Does the lexical expression clearly intended by `answer` naturally and semantically fit the situation in `question`? Accept a different expression if it genuinely fits. Ignore only clear spelling errors; do not otherwise repair or reinterpret the answer.",
          "criteria": {
            "true": "The intended lexical expression naturally and accurately fits the situation.",
            "false": "The intended lexical expression does not naturally and accurately fit the situation."
          }
        },

        "question_type == 'cloze_to_word'": {
          "type": "noul",
          "instructions": "Would inserting the lexical expression intended by `answer` into the blank in `question` produce a natural, grammatical completion with the intended meaning and construction? Correct only a clear spelling-only error for this judgment. Do not repair inflection, derivation, particles, word order, or grammar.",
          "criteria": {
            "true": "The intended lexical expression produces a valid completion.",
            "false": "The intended lexical expression does not produce a valid completion."
          }
        },

        "$default": {
          "type": "noul",
          "instructions": "Does the lexical expression clearly intended by `answer` correctly satisfy `question`? Accept a different lexical expression when it genuinely satisfies the question. Ignore only clear spelling errors; do not otherwise repair the answer.",
          "criteria": {
            "true": "The intended lexical expression correctly satisfies the question.",
            "false": "The intended lexical expression does not correctly satisfy the question."
          }
        }
      }
    },

    "spelling_error": {
      "type": "noul",
      "instructions": "Does `answer` contain a clear spelling-only error in an otherwise recognizable intended lexical expression? Judge orthography only. Do not consider whether the intended expression correctly answers `question`. Do not count inflection, derivation, missing or incorrect particles, word order, or grammar as spelling errors.",
      "criteria": {
        "true": "A recognizable intended lexical expression contains a clear spelling-only error.",
        "false": "There is no clear spelling-only error."
      }
    }
  }
}

// Chốt design mới cho **single-word grading** như sau. Raw output cuối cùng của grader chỉ có đúng 3 field:

// ```text
// task_fit: bool
// spelling_error: bool
// sense_id: string | null
// ```

// Không còn `verdict: correct | typo | incorrect` trong Jev. Consumer muốn enum cũ thì tự derive.

// ## Template 1 — Grade answer

// ```json
// {
//   "state": {
//     "question": {
//       "$eval": "question"
//     },
//     "answer": {
//       "$eval": "answer"
//     }
//   },

//   "questions": {
//     "task_fit": {
//       "$switch": {
//         "question_type == 'definition_to_word'": {
//           "type": "noul",
//           "instructions": "Does the lexical expression clearly intended by `answer` accurately satisfy the definition in `question`? Accept a different lexical expression if it fully expresses the defined meaning. Ignore only clear spelling errors; do not repair other lexical or grammatical differences.",
//           "criteria": {
//             "true": "The intended lexical expression accurately satisfies the definition.",
//             "false": "The intended lexical expression does not accurately satisfy the definition."
//           }
//         },

//         "question_type == 'context_to_word'": {
//           "type": "noul",
//           "instructions": "Does the lexical expression clearly intended by `answer` naturally and semantically fit the situation in `question`? Accept a different expression if it genuinely fits. Ignore only clear spelling errors; do not otherwise repair or reinterpret the answer.",
//           "criteria": {
//             "true": "The intended lexical expression naturally and accurately fits the situation.",
//             "false": "The intended lexical expression does not naturally and accurately fit the situation."
//           }
//         },

//         "question_type == 'cloze_to_word'": {
//           "type": "noul",
//           "instructions": "Would inserting the lexical expression intended by `answer` into the blank in `question` produce a natural, grammatical completion with the intended meaning and construction? Correct only a clear spelling-only error for this judgment. Do not repair inflection, derivation, particles, word order, or grammar.",
//           "criteria": {
//             "true": "The intended lexical expression produces a valid completion.",
//             "false": "The intended lexical expression does not produce a valid completion."
//           }
//         },

//         "$default": {
//           "type": "noul",
//           "instructions": "Does the lexical expression clearly intended by `answer` correctly satisfy `question`? Accept a different lexical expression when it genuinely satisfies the question. Ignore only clear spelling errors; do not otherwise repair the answer.",
//           "criteria": {
//             "true": "The intended lexical expression correctly satisfies the question.",
//             "false": "The intended lexical expression does not correctly satisfy the question."
//           }
//         }
//       }
//     },

//     "spelling_error": {
//       "type": "noul",
//       "instructions": "Does `answer` contain a clear spelling-only error in an otherwise recognizable intended lexical expression? Judge orthography only. Do not consider whether the intended expression correctly answers `question`. Do not count inflection, derivation, missing or incorrect particles, word order, or grammar as spelling errors.",
//       "criteria": {
//         "true": "A recognizable intended lexical expression contains a clear spelling-only error.",
//         "false": "There is no clear spelling-only error."
//       }
//     }
//   }
// }
// ```

// Hai field này cố tình độc lập về semantics:

// ```text
// task_fit
// = expression mà user định viết có giải đúng bài hay không?

// spelling_error
// = expression đó có spelling-only error hay không?
// ```

// Ví dụ:

// ```text
// question: "an animal that barks"
// answer: "dgo"

// task_fit       = true
// spelling_error = true
// ```

// Nhưng:

// ```text
// question: "producing a lot of light"
// answer: "dgo"

// task_fit       = false
// spelling_error = true
// ```

// Không còn ép hai trường hợp này vào cùng một enum.

// ---

// # Template 2 — Resolve sense

// **Chỉ chạy khi:**

// ```text
// task_fit == true
// AND
// spelling_error == false
// ```

// Backend search chính lexical expression user nhập:

// ```text
// search(answer)
// → top-1 Word
// ```

// Ranking phải:

// ```text
// exact lemma
// exact alias/form
// normalized exact
// >>>>>>>> fuzzy match
// ```

// Sau đó đưa **toàn bộ senses của đúng 1 Word đó** vào Jev.

// ```json
// {
//   "state": {
//     "question": {
//       "$eval": "question"
//     },
//     "answer": {
//       "$eval": "answer"
//     }
//   },

//   "questions": {
//     "matched_sense": {
//       "type": "choice",

//       "instructions": "Which listed meaning of `answer` best matches the meaning that `answer` expresses in `question`? Choose `no_candidate` if none of the listed meanings fits.",

//       "criteria": {
//         "$reduce": {
//           "$eval": "matched_word.senses"
//         },

//         "initial": {
//           "no_candidate": "None of the listed meanings matches the meaning expressed by the answer in the question."
//         },

//         "each(acc, sense)": {
//           "$merge": [
//             {
//               "$eval": "acc"
//             },
//             {
//               "sense_${sense.id}": "${matched_word.lemma} - ${sense.pos} - ${sense.definition}"
//             }
//           ]
//         }
//       }
//     }
//   }
// }
// ```

// Backend convert:

// ```text
// sense_123
// → sense_id = 123

// no_candidate
// → sense_id = null
// ```

// Quan trọng: **sense matching tuyệt đối không quyết định `task_fit`**.

// Nếu search top-1 sai hoặc dictionary của ta thiếu sense:

// ```text
// task_fit = true
// sense_id = null
// ```

// vẫn hoàn toàn hợp lệ.

// ---

// # Pipeline cuối cùng

// ```text
//                     user_answer
//                          │
//                          ▼
//                  ┌───────────────┐
//                  │ Jev Query #1  │
//                  │               │
//                  │ task_fit      │
//                  │ spelling_error│
//                  └───────┬───────┘
//                          │
//              ┌───────────┴────────────┐
//              │                        │
//        task_fit = false         task_fit = true
//              │                        │
//              ▼                ┌───────┴────────┐
//       sense_id = null         │                │
//                         spelling=true    spelling=false
//                               │                │
//                               ▼                ▼
//                        sense_id=null     search(answer)
//                                               │
//                                               ▼
//                                          top-1 Word
//                                               │
//                                               ▼
//                                       all Word.senses
//                                               │
//                                               ▼
//                                        ┌─────────────┐
//                                        │Jev Query #2 │
//                                        │matched_sense│
//                                        └──────┬──────┘
//                                               │
//                                   ┌───────────┴──────────┐
//                                   │                      │
//                               sense_X              no_candidate
//                                   │                      │
//                                   ▼                      ▼
//                            sense_id = X           sense_id = null
// ```

// ## Final output

// Luôn luôn cùng một schema:

// ```json
// {
//   "task_fit": true,
//   "spelling_error": false,
//   "sense_id": "123"
// }
// ```

// Hoặc:

// ```json
// {
//   "task_fit": true,
//   "spelling_error": true,
//   "sense_id": null
// }
// ```

// Hoặc:

// ```json
// {
//   "task_fit": false,
//   "spelling_error": false,
//   "sense_id": null
// }
// ```

// Consumer nếu vẫn cần policy cũ thì derive:

// ```text
// task_fit = false
// → incorrect

// task_fit = true && spelling_error = true
// → typo

// task_fit = true && spelling_error = false
// → correct
// ```

// Còn `sense_id` hoàn toàn độc lập với correctness và chỉ dùng khi ta **map được lexical knowledge cụ thể để tracking**.

// Đây là version em sẽ chốt: **2 Jev queries, 1 database top-1 search, 3 output fields, không trộn spelling với correctness và không để sense matching ảnh hưởng correctness.**
