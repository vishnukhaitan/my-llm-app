# Naive RAG KPI snapshot (A7)

Ran `data/golden_questions.jsonl` through `ask_rag` (k=3) on `data/embeddings.json`.

Index source ids: code_of_conduct, data_retention, expense_policy, it_onboarding, leave_policy, llm_assistant_usage, meeting_rooms, parental_leave, security_passwords, sick_leave, vpn_setup, wfh_policy

## Headline numbers

| Metric | Value |
|---|---|
| Questions | 12 |
| Retrieval hits | 12 / 12 |
| Retrieval hit rate | 100% |
| Mean chat cost / query | $0.000115 |
| Latency p50 | 3782 ms |

Hit means the expected document id appears in the retrieved chunk ids (`leave_policy#0` counts for `leave_policy`).
Cost is the chat completion only. Embedding the question is a separate call and is not included.

## Per question

| id | expected | hit | top sources | ms | cost_usd |
|---|---|---|---|---|---|
| q01 | leave_policy | yes | leave_policy#0, sick_leave#0, parental_leave#0 | 3602 | 0.000105 |
| q02 | parental_leave | yes | parental_leave#0, leave_policy#0, leave_policy#1 | 3479 | 0.000100 |
| q03 | meeting_rooms | yes | meeting_rooms#0, meeting_rooms#1, wfh_policy#1 | 4135 | 0.000130 |
| q04 | vpn_setup | yes | vpn_setup#0, vpn_setup#1, wfh_policy#1 | 3843 | 0.000113 |
| q05 | sick_leave | yes | sick_leave#0, sick_leave#1, parental_leave#0 | 4545 | 0.000151 |
| q06 | wfh_policy | yes | wfh_policy#0, sick_leave#2, wfh_policy#1 | 3889 | 0.000121 |
| q07 | expense_policy | yes | expense_policy#0, expense_policy#1, sick_leave#0 | 3970 | 0.000128 |
| q08 | security_passwords | yes | security_passwords#0, llm_assistant_usage#1, it_onboarding#1 | 3720 | 0.000102 |
| q09 | data_retention | yes | data_retention#0, llm_assistant_usage#0, llm_assistant_usage#1 | 3156 | 0.000098 |
| q10 | code_of_conduct | yes | code_of_conduct#0, sick_leave#2, parental_leave#1 | 3614 | 0.000105 |
| q11 | it_onboarding | yes | it_onboarding#0, it_onboarding#1, parental_leave#1 | 4091 | 0.000127 |
| q12 | llm_assistant_usage | yes | llm_assistant_usage#0, llm_assistant_usage#1, data_retention#0 | 3366 | 0.000102 |
