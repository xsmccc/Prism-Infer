# P7-000: P6.12 冻结与 claim 校正

- 状态: `RESOLVED`
- 冻结 commit: `c970c61`
- tag: `p6.12-content-aware-kv`
- 硬件: NVIDIA GeForce RTX 5090 32GB

## 现象

`docs/ROADMAP.md` 顶部仍写着“首个合格策略仍未找到”，但 P6.12-C 已在后文
记录 last-layer attention scorer 通过 reference task gate。这会导致同一份项目
文档对当前能力给出相反结论。

同时，本地 `main` 比 `origin/main` ahead 2。如果继续开始 P7 benchmark，raw
record 中的 commit 可能无法由外部仓库取得，降低复现性。

## 如何发现

开始 P7 前同时检查：

```bash
git status --short --branch
git log --oneline --decorate -n 8
sed -n '1,80p' docs/ROADMAP.md
```

版本状态和文档状态必须一起审计。只看后文最新实验容易遗漏顶部摘要陈旧。

## 解决方案

1. 推送 `e51c16d` 和 `c970c61` 到 `origin/main`。
2. 在 `c970c61` 创建并推送 annotated tag `p6.12-content-aware-kv`。
3. 更新 ROADMAP 顶部状态和独立 claim ledger。
4. 将 P6.2-B hardware counter、P6.8 两卡实测和 FP8 quality 保留为未完成，
   不因 BF16 content-aware 路径完成而隐去。

## 为什么有效

tag 将 P6.12 的代码、测试和质量/性能证据绑定到不可歧义的 commit。claim
ledger 把“机制完成”“质量通过”“性能占优”拆开，避免一个 PASS 被错误外推到
其他维度。

## 验证

```bash
git status --short --branch
git show --no-patch p6.12-content-aware-kv
git ls-remote --tags origin p6.12-content-aware-kv
```

## 剩余限制

- P6.12 的质量结论是 7-image lexical reference gate，不是完整 COCO CIDEr/SPICE。
- BF16 physical compaction 已通过该门禁；FP8 quality 尚未通过。
- P6.2-B 和 P6.8 仍需要合适的外部硬件环境。

## 面试表达

> 我把每个性能里程碑绑定到 clean commit 和可推送 tag，并维护正向/禁止 claim，
> 因为推理系统很容易把局部显存或 kernel 结果误写成端到端吞吐结论。
