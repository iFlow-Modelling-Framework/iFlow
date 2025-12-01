import importlib.util
import importlib.machinery
import sys
import os
from src.util.diagnostics import KnownError
from src.util.localpath import localpath


def dynamicImport(path, name):
    # Normalize dotted path
    if not path:
        packagePath = ""
    else:
        packagePath = path.replace("\\", ".").replace("/", ".")
        if not packagePath.endswith("."):
            packagePath += "."

    full_module_name = packagePath + name
    workpath = full_module_name.split(".")

    # ---------------------------------------------------------------
    # 1. Locate the actual .py file
    # ---------------------------------------------------------------
    module_file_path = None
    for search_dir in localpath:
        candidate = os.path.join(search_dir, *workpath) + ".py"
        if os.path.exists(candidate):
            module_file_path = candidate
            break

    if module_file_path is None:
        raise KnownError(f"Could not find {name}.py in path {path}")

    # ---------------------------------------------------------------
    # 2. Create ALL package levels (turbulence2DV, turbulence2DV.util, ...)
    # ---------------------------------------------------------------
    for i in range(len(workpath) - 1):

        pkg_name = ".".join(workpath[:i + 1])

        if pkg_name not in sys.modules:
            spec = importlib.machinery.ModuleSpec(
                name=pkg_name,
                loader=None,
                is_package=True
            )
            pkg = importlib.util.module_from_spec(spec)

            # Create path for this package level
            for base in localpath:
                pkg_path = os.path.join(base, *workpath[:i + 1])
                if os.path.isdir(pkg_path):
                    pkg.__path__ = [pkg_path]
                    break
            else:
                pkg.__path__ = []

            sys.modules[pkg_name] = pkg

    # ---------------------------------------------------------------
    # 3. Now load the actual module file
    # ---------------------------------------------------------------
    try:
        spec = importlib.util.spec_from_file_location(full_module_name, module_file_path)
        module = importlib.util.module_from_spec(spec)

        # Needed for relative imports
        module.__package__ = ".".join(workpath[:-1])
        module.__file__ = module_file_path

        # Register before exec so relative imports can see it
        sys.modules[full_module_name] = module

        spec.loader.exec_module(module)

    except Exception as e:
        raise KnownError(f"Failed to load module '{full_module_name}'.", e)

    # ---------------------------------------------------------------
    # 4. Return class/function with the same name
    # ---------------------------------------------------------------
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise KnownError(
            f"Class '{name}' not found inside module '{full_module_name}'.",
            e
        )
