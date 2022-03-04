import os
import io
import json
import oss2
import cv2
import configparser
import enum
import numpy as np

from PIL import Image
from typing import Any, Union, Dict, List, Tuple
from multiprocessing import Queue, Process, Pool
from loguru import logger


class StoreType(enum.Enum):

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, type_name: str):
        self.type_name = type_name

    LOCAL = "local"
    OSS = "oss"


class DataType(enum.Enum):

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, type_name: str):
        self.type_name = type_name

    BYTES = "bytes"
    IMAGE = "image"
    NUMPY = "numpy"
    JSON = "json"
    # do nothing, just return input
    DUMMY = "dummy"


def read_oss_config(path: str) -> Dict[str, Any]:
    oss_src_config = configparser.ConfigParser()
    oss_src_config.read(os.path.expanduser(path))
    return oss_src_config['Credentials']


def create_oss_bucket(oss_config: Union[Dict[str, str], str]) -> oss2.Bucket:
    if isinstance(oss_config, str):
        oss_config = read_oss_config(oss_config)

    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])

    return oss2.Bucket(auth, endpoint=oss_config['endpoint'], bucket_name=oss_config['bucket'])


class Reader(object):
    def read(self, path):
        raise NotImplementedError


def oss_read(bucket, key, oss_root=None):
    path = os.path.join(oss_root, key) if oss_root else key
    data_bytes = bucket.get_object(path).read()
    return data_bytes


class OssReader(Reader):
    def __init__(self, oss_config: str):
        self.bucket = create_oss_bucket(oss_config)

    def read(self, path):
        return oss_read(self.bucket, path)


class LocalReader(Reader):
    def read(self, path):
        return open(path, 'rb').read()


class DummyReader(Reader):
    def __init__(self, reader: Reader):
        self.reader = reader

    def read(self, path):
        return path


class BytesReader(Reader):
    def __init__(self, reader: Reader):
        self.reader = reader

    def read(self, path):
        return self.reader.read(path)


class ImageReader(Reader):
    def __init__(self, reader: Reader):
        self.reader = reader

    def read(self, path) -> np.ndarray:
        return Image.open(io.BytesIO(self.reader.read(path)))


class NumpyReader(Reader):
    def __init__(self, reader: Reader):
        self.reader = reader

    def read(self, path) -> Union[np.ndarray, dict]:
        if path.endswith("npz"):
            with np.load(io.BytesIO(self.reader.read(path))) as data:
                return dict(data)
        return np.load(io.BytesIO(self.reader.read(path)))


class JsonReader(Reader):
    def __init__(self, reader: Reader):
        self.reader = reader

    def read(self, path) -> Union[np.ndarray, dict]:
        return json.load(io.BytesIO(self.reader.read(path)))


def build_reader(store_type: str, data_type: str, **kwargs) -> Reader:
    if store_type == StoreType.LOCAL.type_name:
        reader = LocalReader()
    elif store_type == StoreType.OSS.type_name:
        reader = OssReader(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    if data_type == DataType.BYTES.type_name:
        return BytesReader(reader)
    elif data_type == DataType.IMAGE.type_name:
        return ImageReader(reader)
    elif data_type == DataType.NUMPY.type_name:
        return NumpyReader(reader)
    elif data_type == DataType.JSON.type_name:
        return JsonReader(reader)
    elif data_type == DataType.DUMMY.type_name:
        return DummyReader(reader)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


class Writer(object):
    def write(self, path: str, data: Any):
        raise NotImplementedError


class OssWriter(Writer):
    def __init__(self, oss_config: str):
        self.bucket = create_oss_bucket(oss_config)

    def write(self, path, data: bytes):
        return self.bucket.put_object(path, data)


class LocalWriter(Writer):
    def write(self, path, obj: bytes):
        return open(path, 'wb').write(obj)


class BytesWriter(Writer):
    def __init__(self, writer: Writer):
        self.writer = writer

    def write(self, path: str, data: Union[bytes, str]):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self.writer.write(path, data)


class ImageWriter(Writer):
    def __init__(self, writer: Writer):
        self.writer = writer

    def write(self, path: str, data: np.ndarray):
        ext = os.path.splitext(path)[-1]
        ret, img = cv2.imencode(ext, data)
        return self.writer.write(path, img.tobytes())


class NumpyWriter(Writer):
    def __init__(self, writer: Writer):
        self.writer = writer

    def write(self, path:str, data: Union[np.ndarray, dict]):
        output = io.BytesIO()

        if path.endswith("npz"):
            if isinstance(data, list):
                np.savez(output, *data)
            elif isinstance(data, dict):
                np.savez(output, **data)
            else:
                raise ValueError('invalid type: {} to save to {}', type(data), path)
        else:
            if isinstance(data, np.ndarray):
                np.save(output, data)
            else:
                raise ValueError('invalid type: {} to save to {}', type(data), path)
        output = output.getvalue()

        return self.writer.write(path, output)


class JsonWriter(Writer):
    def __init__(self, writer: Writer):
        self.writer = writer

    def write(self, path: str, data: Union[List, Dict, bytes]):
        if isinstance(data, list) or isinstance(data, dict):
            output = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
        elif isinstance(data, bytes):
            output = data
        elif isinstance(data, str):
            output = data.encode('utf-8')
        else:
            raise ValueError('invalid type: {} to save to {}', type(data), path)

        return self.writer.write(path, output)


def build_writer(store_type: str, data_type: str, **kwargs) -> Writer:
    if store_type == StoreType.LOCAL.type_name:
        writer = LocalWriter()
    elif store_type == StoreType.OSS.type_name:
        writer = OssWriter(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    if data_type == DataType.BYTES.type_name:
        return BytesWriter(writer)
    elif data_type == DataType.IMAGE.type_name:
        return ImageWriter(writer)
    elif data_type == DataType.NUMPY.type_name:
        return NumpyWriter(writer)
    elif data_type == DataType.JSON.type_name:
        return JsonWriter(writer)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


class AsyncWriter(object):
    def __init__(self, pool_size: int, store_type: str, data_type: str, **config):
        self.pool_size = pool_size
        self.writer = build_writer(store_type=store_type, data_type=data_type, **config)

        self.in_queue = Queue()
        self.eof_sig = [None, None]

        def worker_loop(writer: Writer, in_queue: Queue):
            while True:
                path, data = in_queue.get()

                if path is None and data is None:
                    logger.info("Finish processing, exit...")
                    break

                writer.write(path, data)

        self.workers = []
        for _ in range(self.pool_size):
            p = Process(target=worker_loop, args=(self.writer, self.in_queue))
            p.start()
            self.workers.append(p)

    def consume(self, data: Tuple[str, Any]):
        self.in_queue.put(data)

    def stop(self):
        for _ in range(self.pool_size):
            self.in_queue.put(self.eof_sig)

        for p in self.workers:
            p.join()
