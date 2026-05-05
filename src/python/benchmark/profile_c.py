#!/usr/bin/env python3
"""Legacy profiler.

El perfilador antiguo dependía de volcados C (`DUMP_Y_PRE`, `DUMP_M`,
`DUMP_Y_FLOAT`, `DUMP_STAGES`) que ya no se generan en la ruta optimizada del
encoder. Se conserva como marcador histórico para evitar usar resultados
incomparables con la base actual.
"""

import sys


def main() -> int:
    print(
        "profile_c.py es legacy: los dumps C que necesita no forman parte de la "
        "ruta optimizada actual. Usa src/python/benchmark/bench_c.py para medir "
        "el encoder C y src/python/analysis/validate_e2e.py para validar la "
        "reconstruccion.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
