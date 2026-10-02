# Ancient-DNA complementary-haplotype phasing API

从含降解错配的分子读段联合复原一对互补二值单倍型，并为每条读段给出唯一归属。
逐位多数表决会把两条同源染色体的证据拼成不存在的序列；本服务对所有规范候选
单倍型（首位固定为 0，商掉"交换两组"的等价性）联合优化：

1. **总错配代价最小**（逐位正整数代价求和）；
2. 再最小化**单条读段最大错配位点数**；
3. 再按（单倍型, 逐读段归属）字典序定唯一解，或稳定地判定歧义。

每条读段的错配位点数不得超过其允许值，且两组各至少两条读段。并列时返回
字典序最前的两个不同解；无解或读段区间未连续覆盖全部位点时返回明确业务码。

## 污染读段处理（可选）

古 DNA 提取物可能混入环境来源分子。请求中可**选填** `max_contaminant_reads`
（整数 1–4），并为**每条读段**提供正整数 `contaminant_penalty`：

- 启用后每条读段在三个状态间**联合**选择：单倍型组（0）、互补组（1）或
  **污染状态（2）**——不是先按旧算法求解再事后剔除异常读段；
- 污染读段**不受**该读段 `max_mismatches` 错配上限约束（罚分即其代价），
  但污染读段总数不得超过 `max_contaminant_reads`；
- 两个同源组各仍须至少包含 **2 条非污染读段**；
- 优化目标依次为：① 非污染读段错配代价 + 污染罚分之和最小；
  ② 非污染读段最大错配数最小；③ 按（规范单倍型, 三向归属，标签
  0 < 1 < 2）字典序稳定判定唯一或歧义，等优时仍只返回排序最前的两个完整解释；
- 响应逐条返回错配证据、罚分以及该读段被判为污染的原因证据
  （相对两组的错配数/代价/位点、可行组、最优 cap 内最便宜选项等）。

`max_contaminant_reads` 与 `contaminant_penalty` **均不出现**时，请求、
响应、裁决与错误行为与旧版完全一致（响应中不含任何污染模式字段）。
二者必须同时、完整出现（每个读段都要有正整数罚分），否则返回
`INVALID_INPUT`；只设置其中之一同样视为非法输入。

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
- `contaminant_penalty`（正整数）为**可选**字段：仅当请求设置
  `max_contaminant_reads` 时必须为每条读段提供，否则不得出现。
- `max_contaminant_reads`（顶层，选填）：1–4 的整数；启用三向污染模式。

**成功响应（200，未启用污染模式）**：

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

### 污染模式响应差异

启用 `max_contaminant_reads` 时，`data` 额外回显 `max_contaminant_reads`；
每个 `solution` 额外包含：

| 字段 | 含义 |
| --- | --- |
| `groups.contaminants` / `contaminant_reads` | 被判为污染的读段 `id`（≤ cap） |
| `contaminant_count` | 污染读段数量 |
| `total_contaminant_penalty` | 污染罚分之和 |
| `total_objective_cost` | 首要目标值 = `total_mismatch_cost` + `total_contaminant_penalty` |
| `total_mismatch_cost` / `max_per_read_mismatches` | **仅统计非污染读段** |
| `assignments` | 三向标签：0=单倍型组，1=互补组，2=污染 |

`per_read` 中：

- 非污染读段：`group` 为 0/1，`status` 为 `haplotype`/`complement`，
  `contaminant=false`，`contaminant_penalty=0`，错配三字段照常给出；
- 污染读段：`group=null`、`status="contaminant"`、`contaminant=true`，
  错配三字段为 `null`，`contaminant_penalty` 为该读段罚分，并给出
  `contaminant_reason`：

```json
{
  "id": "c0",
  "group": null,
  "status": "contaminant",
  "contaminant": true,
  "contaminant_penalty": 5,
  "contaminant_reason": {
    "allowed_mismatches": 0,
    "optimal_max_mismatches": 0,
    "feasible_groups": [],
    "mismatches_vs_haplotype":  { "count": 2, "cost": 18, "positions": [0, 1] },
    "mismatches_vs_complement": { "count": 2, "cost": 18, "positions": [2, 3] },
    "cheapest_allowance_compliant_option": null,
    "cheapest_within_cap_option": null,
    "explanation": "exceeds max_mismatches=0 against both ... (人类可读原因)"
  },
  "mismatch_count": null,
  "mismatch_cost": null,
  "mismatch_positions": null
}
```

`mismatches_vs_*` 是该污染读段相对两组的**完整对照证据**（不因污染状态而
省略）；`feasible_groups` 列出按其自身错配上限可加入的组；
`cheapest_allowance_compliant_option` / `cheapest_within_cap_option` 分别给出
按读段上限、按本解最优 cap 可加入的最便宜一侧（无则 `null`），
`explanation` 说明联合优化为何选择污染（双侧均超限 / 低于最优 cap 内代价 /
维持次目标 cap 等）。污染模式的歧义说明 `note` 也改用三向表述。

**业务错误**：

| HTTP | `error.code` | 触发 |
| --- | --- | --- |
| 400 | `BAD_JSON` | 请求体不是合法 JSON |
| 422 | `INVALID_INPUT` | 数量/形状/取值不合规（含读段区间倒置、代价非正、`max_contaminant_reads` 越界、缺罚分等） |
| 409 | `DISCONTINUOUS_INPUT` | 读段自身不连续或读段并集未覆盖全部位点 |
| 409 | `NO_SOLUTION` | 未启用污染模式时，在允许错配数与"两组各≥2"约束下无可行联合解 |
| 409 | `INSUFFICIENT_CONTAMINANT_CAPACITY` | 污染模式：每个规范单倍型都迫使超过 cap 条读段在双侧超限，污染名额不足 |
| 409 | `INSUFFICIENT_GROUP_EVIDENCE` | 污染模式：污染名额能容纳全部强制污染读段，但不存在每个同源组各 ≥2 条可行非污染读段的候选 |

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
`pytest` 代码测试与针对**含错配样例**和**歧义样例**（以及污染三向定相、
污染名额业务错误、污染选项校验、旧 API 字段不变、无解、不连续、
非法输入）的 API 冒烟，随后自行退出。

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

**污染模式（三向联合 DP）**：

- 候选表与旧路径相同；先用仅基于可行性的**必要条件预筛**剔除无望候选
  （强制污染读段数 ≤ q，且可行读段能让两组各 ≥2；这只是必要条件，最终
  裁决仍由精确 DP 给出）；
- 对保留候选做向量化三向 DP：状态为（组0计数 × 污染计数 ≤ q），每条读段
  三条边——组0/组1 要求满足读段错配上限、污染边**不受错配上限约束**、代价
  为该读段罚分；先算 ① 非污染错配代价 + 罚分的全局最小；
- 再对并列候选从 0 起递增非污染读段错配数 cap K（同样三向 DP），首次可行
  即次目标最优；候选按字典序取前两个；
- 最后对至多两个候选跑精确三向 DP（以 3 进制编码 0/1/2，高读段号在先，
  整数序即三向字典序；状态含组0计数、污染计数、两组最大错配数），返回
  字典序最前的一或两个完整解释及逐条污染原因证据；
- 无解时依据预筛信息区分"污染名额不足"与"两组有效证据不足"两个业务码。

18 位点 × 36 读段的最坏输入在普通硬件上：旧路径约 2–3 秒；污染模式实际
数据因预筛通常 < 1 秒，无任何预筛可借力的对抗输入（全部读段双侧可行）
约 4 秒。
