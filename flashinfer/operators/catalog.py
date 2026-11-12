"""Installed logical operator definitions; implementations have a separate catalog."""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

from flashinfer.runtime import SchemaError
from .definition import OperatorDefinition


@lru_cache(maxsize=1)
def entries():
    path = Path(__file__).resolve().parents[1] / 'data/operators.json'
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding='utf-8'))
    if set(data) != {'schema_version', 'operators'} or data['schema_version'] != 2:
        raise SchemaError('invalid installed logical operator catalog')
    records = tuple(data['operators'])
    identities = set()
    for record in records:
        if (set(record) != {'name', 'domain', 'op', 'module', 'export', 'cases', 'schema', 'output_names', 'description', 'runtime_files'}
                or record['op'] in identities
                or record['module'] != 'flashinfer.operators.' + record['domain'] + '.' + record['name'] + '.operator'
                or not isinstance(record['runtime_files'], dict)
                or not isinstance(record['export'], bool) or not isinstance(record['cases'], bool)):
            raise SchemaError('invalid or duplicate logical operator record')
        identities.add(record['op'])
    return records


def _record(name):
    records = [r for r in entries() if name in (r['name'], r['op'])]
    if len(records) != 1:
        raise SchemaError('no unique installed operator declaration: ' + name)
    return records[0]


@lru_cache(maxsize=None)
def definition(name):
    record = _record(name)
    package = Path(__file__).resolve().parents[1]
    for relative, digest in record['runtime_files'].items():
        path = package / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(package.resolve())
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise SchemaError('installed operator hook digest mismatch: ' + relative)
    result = importlib.import_module(record['module']).OPERATOR
    if (not isinstance(result, OperatorDefinition)
            or (result.name, result.domain, result.op) != tuple(record[k] for k in ('name', 'domain', 'op'))
            or result.schema != record['schema'] or list(result.output_names) != record['output_names']):
        raise SchemaError('installed operator identity differs from its catalog')
    return result


def for_kernel(kernel_id):
    from .implementation import implementation_entries
    record = next((r for r in implementation_entries() if r['kernel_id'] == kernel_id), None)
    return None if record is None else definition(record['op'])


def cases_module(name):
    record = _record(name)
    definition(record['op'])
    if not record['cases']:
        raise SchemaError('operator has no generic numerical case provider: ' + name)
    return importlib.import_module(record['module'].removesuffix('.operator') + '.cases')


def install_exports(namespace, *, domain=None):
    from .runtime import public_function
    for record in entries():
        if not record['export'] or domain is not None and record['domain'] != domain:
            continue
        name = record['name']
        if name in namespace:
            raise SchemaError('operator export collides with an existing API: ' + name)
        namespace[name] = public_function(record['op'])
        if '__all__' in namespace:
            namespace['__all__'].append(name)
