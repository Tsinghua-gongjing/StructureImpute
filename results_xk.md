

# Base model

Base model:  train on icSHAPE hek293 vivo?


# Fine-tuning

## Dataset icSHAPE hek293 vitro

| Mode              | R     | onlyNull  | #GPU | Time     |
| ----------------- | ----- | --------- | ---- | -------- |
| Base              | 0.458 | 0.0842901 | --   | --       |
| FromScratch       | 0.275 | ---       | --   | --       |
| Base+FT           | 0.460 | 0.0617411 | 8    | 00:03:45 |
| Base+FT+DA        | 0.469 | 0.0654491 | 8    | 00:22:31 |
| Base+FT+DA+LR5    | 0.471 | 0.0572981 | 8    | 00:20:06 |

## Dataset icSHAPE hek293 ch

| Mode              | R     | onlyNull  | #GPU | Time     |
| ----------------- | ----- | --------- | ---- | -------- |
| Base              | --    | --        | --   | --       |
| Base+FT           | 0.426 | 0.1850200 | 8    | 00:01:02 |

## Dataset icSHAPE hek293 cy

| Mode              | R     | onlyNull  | #GPU | Time     |
| ----------------- | ----- | --------- | ---- | -------- |
| Base              | --    | --        | --   | --       |
| Base+FT           | 0.494 | 0.1826680 | 8    | 00:01:05 |

## Dataset icSHAPE hek293 np

| Mode              | R     | onlyNull  | #GPU | Time     |
| ----------------- | ----- | --------- | ---- | -------- |
| Base              | --    | --        | --   | --       |
| Base+FT           | 0.510 | 0.1578580 | 8    | 00:03:45 |

### Dataset icSHAPE mes

| Mode             | R     | onlyNull  | #GPU | Time     |
| ---------------- | ----- | --------- | ---- | -------- |
| Base             | 0.458 | 0.2209948 | --   | --       |
| FromScratch      | 0.166 | ---       | --   | --       |
| Base+FT          | 0.163 | 0.2229443 | 8    | 01:20:17 |
| Base+FT+5000/618 | 0.472 | 0.2110634 | 8    | 00:02:51 |


## Dataset DMSseq K562 vivo

| Mode       | R     | maskNull   | #GPU | Time     |
| ---------- | ----- | ---------- | ---- | -------- |
| Base       | --    | --         |      |          |
| Base+FT    | 0.692 | 0.03823998 | 8    | 01:23:11 |
| Base+FT+DA | --    | --         | 8    |          |



## Dataset DMSseq K562 vitro

| Mode       | R     | maskNull   | #GPU | Time     |
| ---------- | ----- | ---------- | ---- | -------- |
| Base       | --    | --         | --   | --       |
| Base+FT    | 0.416 | 0.13352124 | 8    | 00:02:08 |
| Base+FT+DA | --    | --         | 8    |          |

## Dataset DMSseq fibroblast vivo

| Mode       | R     | maskNull   | #GPU | Time     |
| ---------- | ----- | ---------- | ---- | -------- |
| Base       | --    | --         | --   | --       |
| Base+FT    | 0.414 | 0.15294927 | 8    | 00:01:58 |
| Base+FT+DA |       |            | 8    |          |

## Dataset DMSseq fibroblast vitro

| Mode       | R     | maskNull   | #GPU | Time     |
| ---------- | ----- | ---------- | ---- | -------- |
| Base       | --    | --         | --   | --       |
| Base+FT    | 0.681 | 0.04918523 | 8    | 00:35:52 |
| Base+FT+DA | --    | --         | 8    |          |


