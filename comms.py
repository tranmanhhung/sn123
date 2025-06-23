# MIT License
#
# Copyright (c) 2024 MANTIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import json
import asyncio
import botocore
from typing import List, Dict
from aiobotocore.session import get_session
import aiohttp, email.utils
import base64
import logging

R2_BUCKET_ID = None
def bucket():
    global R2_BUCKET_ID
    if R2_BUCKET_ID is not None:
        return R2_BUCKET_ID
    try:
        R2_BUCKET_ID = os.getenv("R2_BUCKET_ID")
        if not R2_BUCKET_ID:
            return None
        return R2_BUCKET_ID
    except Exception as e:
        return None

R2_ACCOUNT_ID = None
def load_r2_account_id():
    global R2_ACCOUNT_ID
    if R2_ACCOUNT_ID is not None:
        return R2_ACCOUNT_ID
    try:
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        if not R2_ACCOUNT_ID:
            return None
        return R2_ACCOUNT_ID
    except Exception as e:
        return None

R2_ENDPOINT_URL = None
def load_r2_endpoint_url():
    global R2_ENDPOINT_URL
    account_id = load_r2_account_id()
    if not account_id:
        return None
    try:
        R2_ENDPOINT_URL = f"https://{account_id}.r2.cloudflarestorage.com"
        return R2_ENDPOINT_URL
    except Exception as e:
        return None

R2_WRITE_ACCESS_KEY_ID = None
def load_r2_write_access_key_id():
    global R2_WRITE_ACCESS_KEY_ID
    if R2_WRITE_ACCESS_KEY_ID is not None:
        return R2_WRITE_ACCESS_KEY_ID
    try:
        R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_WRITE_ACCESS_KEY_ID")
        if not R2_WRITE_ACCESS_KEY_ID:
            return None
        return R2_WRITE_ACCESS_KEY_ID
    except Exception as e:
        return None

R2_WRITE_SECRET_ACCESS_KEY = None
def load_r2_write_secret_access_key():
    global R2_WRITE_SECRET_ACCESS_KEY
    if R2_WRITE_SECRET_ACCESS_KEY is not None:
        return R2_WRITE_SECRET_ACCESS_KEY
    try:
        R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
        if not R2_WRITE_SECRET_ACCESS_KEY:
            return None
        return R2_WRITE_SECRET_ACCESS_KEY
    except Exception as e:
        return None

CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256)
session = get_session()
    
def get_local_path(bucket: str, filename: str ) -> str:
    return os.path.join(os.path.expanduser("~/storage"), bucket, filename)

async def exists_locally( bucket:str, filename: str ) -> bool:
    local_path = get_local_path( bucket, filename )
    return os.path.exists(local_path)

async def delete_locally( bucket:str, filename: str ) -> None:
    local_path = get_local_path( bucket, filename )
    if os.path.exists(local_path):
        try:
            await asyncio.to_thread(os.remove, local_path)
        except Exception as e:
            pass

async def load( bucket:str, filename: str) -> dict:
    local_path = get_local_path(bucket, filename)
    try:
        def load_json(path):
            with open(path, "r") as f:
                return json.load(f)
        data = await asyncio.to_thread(load_json, local_path)
        return data
    except Exception as e:
        return None


async def _local_path_from_url(url: str) -> str:
    """Map https://host/path -> ~/storage/host/path"""
    name = url.split("://", 1)[1]
    return os.path.join(os.path.expanduser("~/storage"), name)

async def download(url: str):

    path = await _local_path_from_url(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=600) as r:
            r.raise_for_status()
            try:
                data = await r.json()
            except Exception:
                try:
                    data_bytes = await r.read()
                    try:
                        data = data_bytes.decode()
                    except Exception:
                        data = data_bytes
                except Exception:
                    data = None

    try:
        if isinstance(data, (dict, list, str, int, float, bool, type(None))):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "wb") as f:
                f.write(data if isinstance(data, (bytes, bytearray)) else bytes(str(data), "utf-8"))
    except Exception:
        pass
    return data




async def exists( bucket:str, filename: str ) -> bool:
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return False

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            await s3_client.head_object(Bucket=bucket, Key=filename)
            return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ["404", "NoSuchKey", "NotFound"]:
            return False
        else:
            return False
    except Exception as e:
        return False
    
    
async def timestamp( bucket:str, filename: str ):
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return None

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            response = await s3_client.head_object(Bucket=bucket, Key=filename)
            last_modified = response.get("LastModified")
            if last_modified is not None:
                return last_modified
            else:
                return None
    except Exception as e:
        return None

async def list( bucket:str, prefix: str ) -> List[str]:
    endpoint_url = load_r2_endpoint_url()
    access_key = load_r2_write_access_key_id()
    secret_key = load_r2_write_secret_access_key()

    if not endpoint_url or not access_key or not secret_key:
        return []

    matching_keys = []    
    try:
        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj.get("Key", "")
                        matching_keys.append(key)
    except Exception as e:
        pass

    return matching_keys


async def timestamp(url: str):
    """Return Last-Modified datetime of a public HTTP object or None."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.head(url, timeout=10) as r:
                if r.status == 200:
                    lm = r.headers.get("Last-Modified")
                    if lm:
                        return email.utils.parsedate_to_datetime(lm)
    except Exception:
        pass
    return None



def _sanitize_b64(obj):
    """Recursively replace any bytes/bytearray with Base64-encoded ASCII str."""
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _sanitize_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_b64(v) for v in obj]
    return obj
