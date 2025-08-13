<!-- GSD:project-start source:PROJECT.md -->
## Project

**oh-my-tai**

极简 LLM CLI 工具。用一句自然语言快速调用你的脚本收藏。

目标用户是有写小脚本习惯的极客玩家——当你忘记某个命令怎么打，或者想调用自己曾经写过的工具时，只需 `tai 帮我做某某事`，无需打开更重的 agent harness。

**Core Value:** **极速响应**。够快才是真需求。功能不必堆砌，够用就行。

### Constraints

- **Tech stack**: Python only，不引入复杂依赖
- **Env management**: uv
- **Config format**: TOML（Python 3.11+ 内置支持）
- **Tool directory**: 用户可配置，默认 `~/.tai/tools/`
- **Execution**: 直接执行，不默认确认（追求速度）
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
