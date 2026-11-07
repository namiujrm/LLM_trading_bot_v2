---
name: add_domain_port
description: >-
  Standardized workflow for defining a new Domain Port interface in Clean Architecture,
  ensuring pure domain dependency, zero default arguments, and contract validation.
---

# Add Domain Port Runbook (`add_domain_port`)

When adding a new Domain Port interface:
1. **Define the Abstract Protocol / ABC in `src/trading_bot/ports/`:**
   - Use `typing.Protocol` or `abc.ABC` with abstract methods.
   - Depend ONLY on Domain entities and standard typing (`src/trading_bot/domain/`).
   - Every method parameter must be explicitly typed with ZERO default values (`def method(self, param: Type) -> ReturnType:`).
2. **Ensure Layer Isolation:**
   - The Port module must not import any adapters, infrastructure, or presentation modules.
3. **Write Contract Verification Test:**
   - Add a structural typing test in `tests/contract/` checking that conforming adapters satisfy the Protocol without runtime instantiation.
4. **Run Architecture Audit:**
   - Run `uv run lint-imports` to verify `ports_depend_only_on_domain` contract is satisfied.
