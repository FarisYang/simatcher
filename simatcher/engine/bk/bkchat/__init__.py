import json
import copy
import logging
from typing import Dict, List, Text, Generator, Union, Optional, Tuple
import urllib

import aiohttp

from simatcher.exceptions import ActionFailed
from simatcher.engine.base import Runner
from simatcher.constants import RANKING, INTENT
from .config import *

logger = logging.getLogger(__name__)

_BACKENDS: List[Tuple[str, str, str]] = []
if BKCHAT_APIGW_ROOT:
    _BACKENDS.append((BKCHAT_APIGW_ROOT, BKCHAT_APP_ID, BKCHAT_APP_SECRET))
if BKCHAT_NEW_APIGW_ROOT:
    _BACKENDS.append((BKCHAT_NEW_APIGW_ROOT, BKCHAT_NEW_APP_ID, BKCHAT_NEW_APP_SECRET))


async def _load_data_from_remote(path: str,
                                 host: str = BKCHAT_APIGW_ROOT,
                                 app_id: str = BKCHAT_APP_ID,
                                 app_secret: str = BKCHAT_APP_SECRET,
                                 method: str = 'POST', **kwargs) -> Dict:
    access_token = urllib.parse.urlencode({'bk_app_code': app_id,
                                           'bk_app_secret': app_secret})
    url = f"{host}/{path}?{access_token}"
    try:
        async with aiohttp.request(method, url, **kwargs) as resp:
            if 200 <= resp.status < 300:
                return json.loads(await resp.text())
            raise ActionFailed(502)
    except aiohttp.InvalidURL:
        raise ActionFailed(401, 'API root url invalid')
    except aiohttp.ClientError:
        raise ActionFailed(403, 'HTTP request failed with client error')


async def _load_data_from_remote_safe(path: str, host: str, app_id: str,
                                      app_secret: str, **kwargs) -> Optional[Dict]:
    """与 _load_data_from_remote 相同，但异常时返回 None 而不抛出。"""
    try:
        return await _load_data_from_remote(path, host, app_id, app_secret, **kwargs)
    except Exception as e:
        logger.warning('_load_data_from_remote_safe: host=%s path=%s err=%s', host, path, e)
        return None


class BKChatEngine:
    def __init__(self, pipeline_config: Dict = BKCHAT_PIPELINE_CONFIG, *args, **kwargs):
        self.pipeline_config = pipeline_config.copy()
        self.runner = Runner.load(self.pipeline_config)

    @classmethod
    async def load_slots(cls, **kwargs) -> List:
        regex_features = []
        for host, app_id, app_secret in _BACKENDS:
            response = await _load_data_from_remote_safe(
                'api/v1/exec/admin_describe_tasks', host, app_id, app_secret, json=kwargs)
            if not response:
                continue
            tasks = response.get('data', [])
            for task in tasks:
                slots = task['slots']
                for slot in slots[::-1]:
                    slot.setdefault('value', '')
                    slot['usage'] = task['index_id']
                    regex_features.append(slot)
        return regex_features

    @classmethod
    async def _load_intents_from_backend(cls, host: str, app_id: str,
                                         app_secret: str, **kwargs) -> Optional[List]:
        response = await _load_data_from_remote_safe(
            'api/v1/exec/admin_describe_intents', host, app_id, app_secret, json=kwargs)
        if not response:
            return None
        db_intents = response.get('data', [])
        if not db_intents:
            return None

        intent_map = {intent['id']: intent for intent in db_intents}

        response = await _load_data_from_remote_safe(
            'api/v1/exec/admin_describe_utterances', host, app_id, app_secret,
            json={'data': {'index_id__in': list(intent_map.keys())}})
        if not response:
            return None

        db_utterances = response.get('data', [])
        utterance_intents = []
        for utterance in db_utterances:
            for sentence in utterance['content']:
                intent = copy.deepcopy(intent_map[utterance['index_id']])
                intent['utterance'] = sentence.lower()
                utterance_intents.append(intent)
        return utterance_intents

    @classmethod
    async def load_corpus_text(cls, **kwargs) -> Union[List, None]:
        all_intents = []
        for host, app_id, app_secret in _BACKENDS:
            intents = await cls._load_intents_from_backend(host, app_id, app_secret, **kwargs)
            if intents:
                all_intents.extend(intents)
        return all_intents or None

    def classify(self,
                 text: Text,
                 pool: List,
                 output_properties: Dict = None,
                 only_output_properties=True,
                 regex_features: List = None):
        if not pool:
            return {}
        output_properties = output_properties or {RANKING, INTENT}
        prune = text.split(' ', maxsplit=1)
        prune[0] = prune[0].lower()
        prune = ' '.join(prune)
        message = self.runner.parse(prune,
                                    output_properties=output_properties,
                                    pool=pool,
                                    text_col='utterance',
                                    regex_features=regex_features)
        return message.as_dict(only_output_properties=only_output_properties)

    def extractor(self):
        pass

    def run(self):
        pass
