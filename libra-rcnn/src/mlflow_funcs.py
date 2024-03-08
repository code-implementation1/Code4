# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MLFlow functions."""

import os
import logging
import argparse
from datetime import datetime
from functools import lru_cache, wraps
from types import ModuleType, FunctionType
from typing import Any, Dict, Union, Optional

from mindspore.communication.management import get_rank


def _get_rank() -> Optional[int]:
    try:
        rank = get_rank()
    except RuntimeError:
        rank = None
    return rank


@lru_cache(maxsize=None)
def mlflow_import() -> Optional[ModuleType]:
    """
    Import MLFlow if possible.
    """
    try:
        import mlflow
    except ImportError:
        mlflow = None
    return mlflow


def mlflow_init() -> None:
    """
    Initialize MLFlow logging.

    To initialize it, you need to set the following environment variables:
    * MLFLOW_TRACKING_URI
    * MLFLOW_EXPERIMENT_NAME

    """
    mlflow = mlflow_import()
    if mlflow is None:
        raise RuntimeError('Cannot import MLFlow.')
    if ('MLFLOW_TRACKING_URI' not in os.environ
            or 'MLFLOW_EXPERIMENT_NAME' not in os.environ):
        raise RuntimeError(
            'Cannot initialize MLFlow logging: '
            'at least one of the required environment variables is not set.'
        )
    if _get_rank() not in [None, 0]:
        logging.warning('MLFlow init: rank > 0; skipping...')
        return
    mlflow.set_experiment(experiment_name=os.environ['MLFLOW_EXPERIMENT_NAME'])
    mlflow.start_run(run_name=datetime.now().isoformat()[:19])
    logging.info('MLFlow was initialized successfully.')


def mlflow_log_args(args: Union[Dict[str, Any]]) -> None:
    """
    Log all arguments as parameters.
    """
    mlflow = mlflow_import()
    if mlflow is None or mlflow.active_run() is None:
        return
    if isinstance(args, dict):
        mlflow.log_params(args)
    elif isinstance(args, argparse.Namespace):
        mlflow.log_params(args.__dict__)
    else:
        raise ValueError(
            f'Cannot log `args` with type `{type(args)}` as MLFlow parameters.'
        )


def mlflow_log_state() -> None:
    """
    Log current code state if possible.
    """
    mlflow = mlflow_import()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        import git
    except ImportError:
        logging.warning('Cannot import GitPython.')
        return
    from git import InvalidGitRepositoryError
    try:
        repo = git.Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        logging.warning('PWD is not a git repository.')
        return
    url = next(repo.remote().urls)
    if '@' in url:
        url = url.split('@')[-1]
    mlflow.set_tags({'git_url': url,
                     'git_branch': repo.active_branch.name,
                     'commit_hash': next(repo.iter_commits()).hexsha})
    if repo.is_dirty():
        mlflow.log_text(repo.git.diff(repo.head.commit.tree), 'git.diff')


def mlflow_decorator(func: FunctionType) -> FunctionType:
    """
    Decorator that initialize MLFlow.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Function wrapper.
        """
        mlflow_init()
        try:
            func(*args, **kwargs)
        finally:
            mlflow_import().end_run()
    return wrapper
