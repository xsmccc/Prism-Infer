# Prism-Infer 文档历史索引

本文件只解释历史文档与当前权威文档的关系，不复制已经过期的阶段计划。

## 当前权威入口

- 当前阶段、完成状态与后续顺序：`docs/ROADMAP.md`
- 可执行验证命令与 PASS 标准：`docs/VERIFICATION.md`
- 性能数字、环境与适用边界：`docs/PERFORMANCE_REPORT.md`
- 可用、受限与禁止的项目结论：`docs/CLAIMS.md`
- 阶段交付格式：`docs/STAGE_DELIVERY_TEMPLATE.md`

这些文件优先于早期日计划、聊天记录和未汇总的实验日志。状态冲突时，以能追溯到
clean commit、命令和 raw evidence 的最新记录为准。

## `DAY_*.md` 状态

`docs/DAY_01.md` 与 `docs/DAY_02.md` 是 P1/P2早期逐日开发记录，已在 commit
`311f055` 删除。它们只保留在 Git历史中，用于考古，不再属于工作树，也不作为当前
需求、完成状态或性能 claim来源。

需要审阅历史内容时可只读查看：

```bash
git show 311f055^:docs/DAY_01.md
git show 311f055^:docs/DAY_02.md
```

禁止从历史日计划中的未完成 checkbox推断当前状态；必须回到 ROADMAP、VERIFICATION
和对应阶段的 clean evidence重新判断。

## 历史结论的使用规则

1. 历史 benchmark只能标明当时 commit、硬件、配置和已知限制，不能替代当前数字。
2. 被后续修复覆盖的 root cause仍可用于问题复盘，但当前行为以最新测试为准。
3. 已删除或 superseded文档不能被 README、简历或面试材料当作完成证据。
4. 任何恢复历史文档的提交都必须加醒目的 `HISTORICAL / NON-AUTHORITATIVE` banner。
