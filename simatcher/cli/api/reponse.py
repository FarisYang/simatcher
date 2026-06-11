import json
import math

import numpy as np


class _SafeEncoder(json.JSONEncoder):
    """处理 numpy 类型和非有限浮点值，避免 JSON 序列化崩溃。"""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            if not math.isfinite(v):
                return None
            return v
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _sanitize(obj):
    """递归将非有限浮点值替换为 None。"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


class Response:
    def __init__(self, result=True, code=0, message='', data={}):
        self.result = result
        self.code = code
        self.data = _sanitize(data)
        self.message = message

    def __iter__(self):
        yield from {
            'result': self.result,
            'code': self.code,
            'data': self.data,
            'message': self.message
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False, cls=_SafeEncoder)

    def __repr__(self):
        return self.__str__()
