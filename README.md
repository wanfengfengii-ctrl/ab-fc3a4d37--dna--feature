# Ancient-DNA complementary-haplotype phasing API

从含降解错配的分子读段联合复原一对互补二值单倍型，并为每条读段给出唯一归属。
逐位多数表决会把两条同源染色体的证据拼成不存在的序列；本服务对所有规范候选
单倍型（首位固定为 0，商掉"交换两组"的等价性）联合优化：

1. **总错配代价最小**（逐位正整数代价求和）；
2. 再最小化**单条读段最大错配位点数**；
3. 再按（单倍型, 逐读段归属）字典序定唯一解，或稳定地判定歧义。

每条读段的错配位点数不得超过其允许值，且两组各至少两条读段。并列时返回
字典序最前的两个不同解；无解或读段区间未连续覆盖全部位点时返回明确业务码。

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
- `max_mismatches` 是该读段允许的错配位点数（非负整数）。

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
| 422 | `INVALID_INPUT` | 数量/形状/取值不合规（含读段区间倒置、代价非正等） |
| 409 | `DISCONTINUOUS_INPUT` | 读段自身不连续或读段并集未覆盖全部位点 |
| 409 | `NO_SOLUTION` | 在允许错配数与"两组各≥2"约束下无可行联合解 |

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
`pytest` 代码测试与针对**含错配样例**和**歧义样例**（以及无解、不连续、
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

18 位点 × 36 读段的最坏输入在普通硬件上约 2–3 秒返回。
