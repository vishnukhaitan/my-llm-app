# MP1 comparison table

From the latest run (gpt-4o-mini extraction, gpt-4o judge, temperature 0). Refresh with `python mp1_prompt_lab.py --refresh`.

| strategy   | Accuracy (mean of 3) | Parse rate | Judge score | Total cost ($) | Latency p50 (s) |
|------------|---------------------:|-----------:|------------:|---------------:|----------------:|
| cot        |                  2.8 |        1.0 |         4.0 |       0.000785 |           3.431 |
| few_shot   |                  2.9 |        1.0 |         3.9 |       0.000493 |           2.208 |
| structured |                  2.9 |        1.0 |         4.0 |       0.000411 |           2.085 |
| zero_shot  |                  2.7 |        1.0 |         3.9 |       0.000322 |           2.108 |

By-field hit rate:

| strategy   | company | role | years |
|------------|--------:|-----:|------:|
| cot        |     0.8 |  1.0 |   1.0 |
| few_shot   |     1.0 |  1.0 |   0.9 |
| structured |     0.9 |  1.0 |   1.0 |
| zero_shot  |     0.8 |  1.0 |   0.9 |

Total extraction spend on this run: about **$0.002**. Judge calls are extra (gpt-4o).
