# Ancient-DNA complementary-haplotype phasing API

从含降解错配的分子读段联合复原一对互补二值单倍型，并为每条读段给出唯一归属。
逐位多数表决会把两条同源染色体的证据拼成不存在的序列；本服务对所有规范候选
单倍型（首位固定为 0，商掉"交换两组"的等价性）联合优化：

1. **总错配代价最小**（逐位正整数代价求和）；
2. 再最小化**单条读段最大错配位点数**；
3. 再按（单倍型, 逐读段归属）字典序定唯一解，或稳定地判定歧义。

每条读段的错配位点数不得超过其允许值，且两组各至少两条读段。并列时返回
字典序最前的两个不同解；无解或读段区间未连续覆盖全部位点时返回明确业务码。

## 污染读段（可选三向归属）

古 DNA 提取物可能混入环境来源分子。调用方可以在原请求中**选填**
`max_contaminant_reads`（1–4 的整数），并为**每条**读段提供正整数
`contaminant_penalty`；两者只要出现其一，服务即进入污染模式，此时每个字段
都必须完整合法。两者均不出现时，请求、响应、裁决与错误行为与旧版完全一致。

启用后，求解器**联合**为每条读段选择：单倍型组（0）、互补组（1）或污染
状态（2）——而不是先按旧算法求解、再事后剔除异常。污染读段：

- 不受 `max_mismatches` 错配上限约束（其错配不计入任何目标）；
- 支付该读段固定的 `contaminant_penalty`；
- 总数不得超过 `max_contaminant_reads`；
- 两条单倍型组仍各至少包含**两条非污染读段**。

目标依次为：

1. **非污染读段错配代价与污染罚分之和最小**；
2. 再最小化**非污染读段的单条最大错配位点数**（污染读段不参与）；
3. 再按规范单倍型及三向归属（0 < 1 < 2）字典序稳定判定唯一或歧义，
   等优时仍只返回排序最前的两个完整解释。

```json
{
  "n_sites": 8,
  "max_contaminant_reads": 1,
  "reads": [
    {
      "id": "r0",
      "start": 0,
      "end": 3,
      "observations": [1, 1, 1],
      "mismatch_costs": [7, 1, 1],
      "max_mismatches": 1,
      "contaminant_penalty": 4
    }
  ]
}
```

污染模式的成功响应在旧字段之外附加：

| 字段 | 含义 |
| --- | --- |
| 顶层 `contaminant_mode` | `{"enabled": true, "max_contaminant_reads": k}` |
| `groups.contaminants` | 被判为污染的读段 `id` 列表 |
| `contaminants` | 每条污染读段的证据：对单倍型/互补的错配数、允许值、罚分、文字原因 |
| `contaminant_count` | 污染读段数量（≤ `max_contaminant_reads`） |
| `total_contaminant_penalty` | 污染罚分合计 |
| `objective_mismatch_cost_plus_penalty` | 第一目标值（非污染错配代价 + 罚分） |
| `per_read[].group` | 取值新增 `2`（污染）；污染读段 `contaminant=true`、错配字段为 `null` |

业务错误新增（均为 409）：

| `error.code` | 触发 |
| --- | --- |
| `INSUFFICIENT_CONTAMINANT_CAPACITY` | 某候选单倍型本可让两组都有足够证据，但需要超过上限的污染读段数量；消息给出至少需要的数量 |
| `INSUFFICIENT_GROUP_EVIDENCE` | 放宽污染数量后仍不存在让两组各有至少两条合规非污染读段的单倍型 |

## 接口

### `POST /api/phase`

```json
{
  "n_sites": 8,
  "reads": [
    {
      "id": "r0",
      "start": 0,
      "end": 3,
      "observations": [1, 1, 1],
      "mismatch_costs": [7, 1, 1],
      "max_mismatches": 1
    }
  ]
}
```

- `n_sites`: 8–18 个有序二值位点；
- `reads`: 10–36 条唯一读段（`id` 不可重复）；
- 每条读段覆盖连续半开区间 `[start, end)`；`observations` 与 `mismatch_costs`
  的长度必须等于 `end - start`，代价为正整数；
- `max_mismatches` 是该读段允许的错配位点数（非负整数）；
- `contaminant_penalty`（可选，正整数）：该读段被判为污染时支付的罚分；
  一旦任意读段携带该字段或请求给出 `max_contaminant_reads`，即进入污染模式，
  两者必须同时完整提供（详见下文）。

**成功响应（200）**：

```json
{
  "ok": true,
  "data": {
    "n_sites": 8,
    "unique": true,
    "solution": { "...": "即 solutions[0]" },
    "solutions": [
      {
        "haplotype": [0, 1, 1, 0, 1, 0, 0, 1],
        "complement": [1, 0, 0, 1, 0, 1, 1, 0],
        "groups": {
          "haplotype": ["r0", "..."],
          "complement": ["r3", "..."]
        },
        "assignments": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "per_read": [
          {
            "id": "r0",
            "group": 0,
            "mismatch_count": 1,
            "mismatch_cost": 7,
            "mismatch_positions": [0]
          }
        ],
        "total_mismatch_cost": 10,
        "max_per_read_mismatches": 1,
        "mismatch_positions": [0, 7]
      }
    ],
    "note": "仅歧义时出现；两个不同最优解已全部返回"
  }
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `haplotype` / `complement` | 规范单倍型（首位为 0）及其逐位补 |
| `groups.haplotype` / `groups.complement` | 两组的读段 `id`（各 ≥ 2） |
| `assignments` | 与输入同序的逐读段归属（0=单倍型组，1=互补组） |
| `per_read` | 每条读段的归属、错配数、错配代价、错配全局位点 |
| `total_mismatch_cost` / `max_per_read_mismatches` | 两个优化目标值 |
| `mismatch_positions` | 整个解涉及的全局错配位（去重排序） |

歧义时 `unique=false` 且 `solutions` 恰有两个元素，按（单倍型, 归属）
字典序排列；唯一时 `solutions` 只有一个。同一单倍型下两条不同最优归属
（仅中性读段去向不同，非组交换等价）也算两个不同解并同样返回。

**业务错误**：

| HTTP | `error.code` | 触发 |
| --- | --- | --- |
| 400 | `BAD_JSON` | 请求体不是合法 JSON |
| 422 | `INVALID_INPUT` | 数量/形状/取值不合规（含读段区间倒置、代价非正、污染字段缺失或越界等） |
| 409 | `DISCONTINUOUS_INPUT` | 读段自身不连续或读段并集未覆盖全部位点 |
| 409 | `NO_SOLUTION` | 旧模式下，在允许错配数与"两组各≥2"约束下无可行联合解 |
| 409 | `INSUFFICIENT_CONTAMINANT_CAPACITY` | 污染模式：需要的污染读段数超过 `max_contaminant_reads` |
| 409 | `INSUFFICIENT_GROUP_EVIDENCE` | 污染模式：两组有效（合规非污染）证据不足 |

```json
{ "ok": false, "error": { "code": "NO_SOLUTION", "message": "..." } }
```

### `GET /health`

返回 `{"status": "healthy"}`，供 Docker 健康检查使用。

## 运行

```bash
# 宿主机端口由 API_PORT 配置
API_PORT=8000 docker compose up --build api
```

容器内置 HEALTHCHECK（`/health`，5 秒间隔，启动宽限 10 秒）。

## 一次性 verify

```bash
API_PORT=8000 bash scripts/verify_all.sh
```

该脚本依次：构建镜像 → 启动 API → 等待容器健康（verify 容器通过
`depends_on: condition: service_healthy` 在服务健康后才启动）→ 运行
`pytest` 代码测试与针对**含错配样例**、**歧义样例**、**污染隔离样例**
（以及污染上限不足、两组证据不足、无解、不连续、非法输入）的 API 冒烟，
随后自行退出。

**汇总退出码**（位掩码）：

| 位 | 含义 |
| --- | --- |
| 1 | 代码测试失败 |
| 2 | API 冒烟失败 |
| 4 | 镜像构建失败 |

全绿退出 0；失败时各位相加指出失败阶段。

也可对已经运行的服务单独执行：

```bash
API_BASE_URL=http://127.0.0.1:8000 python scripts/verify.py
```

## 本地开发

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
.venv/bin/python -m pytest -q
```

## 算法

- numpy 向量化一次性算出所有 `2**(n_sites-1)`（最多 131072）个规范候选对
  每条读段的错配数/代价/可行性；
- 对每个候选做 O(读段数) 贪心分析得到精确最小总代价（满足两组 ≥2 最多只需
  把两条严格偏好读段翻到较贵一侧，等代价中性读段免费补位）；
- 对达到全局最小代价的候选，用状态仅为"组0读段数"的向量化 DP，从 0 起递增
  错配数上界 K，首次可行的 K 即次目标最优；候选与归属均按字典序遍历，取前
  两个不同解；
- 仅对最终至多两个候选跑精确 DP（状态 = 组0计数×两组最大错配数），给出
  字典序最小的逐读段归属与完整错配证据。

**污染模式**不改变候选枚举；三向归属用另一个向量化 DP 求解，状态为
（组0读段数截断于 2、组1读段数截断于 2、污染读段数），饱和组在截断计数上
自环；组边要求合规且错配 ≤ 当前 K，污染边只加罚分、不受错配约束。先以
错配上界完全放松求最小联合目标（错配代价 + 罚分），再对并列候选递增 K，
首个可行的 K 即次目标最优。最终至多两个候选跑一个 k=2 的最佳路径精确 DP
（状态额外记录两组各自最大错配数），给出字典序最小的三向归属与逐读段
错配/罚分/污染原因证据。

18 位点 × 36 读段的最坏输入在普通硬件上约 2–3 秒返回（污染模式状态更小，
实测约 1–2 秒）。
