import argparse
import asyncio
import base64
import configparser
import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
import platform
import posixpath
import re
import shlex
import shutil
import socket
import stat
import subprocess
import tarfile
import tempfile
import time
import uuid
import zipfile
from fnmatch import fnmatch
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urlparse
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr, validator
from starlette.websockets import WebSocketState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIST_DIR = os.path.join(STATIC_DIR, "dist")
DEFAULT_ITASK_DOWNLOAD_URL = "https://scmcenter.cn-hangzhou.alipay.aliyun-inc.com/antcodebuild/artifacts/WK50b0331b2cd74e1c862f92040aacb331/BJ261c41c05b214f3ca301a02df4830aa0/2026-07-02-07-42-02/itask-client/itask-client.zip?Expires=1798530123&OSSAccessKeyId=bWwpuXuj5yuYKDDK&Signature=xtQuKpO2pX8%2BKEFCoCMxOwxf6xc%3D"
ITASK_ARTIFACT_SHA256 = {
    "itask-darwin-arm64": "4183760c128c4c549809140f1d53aaf2db3858953aeb436e5b966b10a508d417",
    "itask-linux-amd64": "1c95d6ea601683ecee7fbd37122a16f2fca741d1323206a2ded0e1e2abd67628",
}
ITASK_INSTALL_DIR = os.path.abspath(os.path.expanduser("~/.local/share/itask-gui/bin"))
ITASK_MANAGED_BIN = os.path.join(ITASK_INSTALL_DIR, "itask")
ITASK_DOWNLOAD_MAX_BYTES = 256 * 1024 * 1024
ITASK_BINARY_MAX_BYTES = 128 * 1024 * 1024
ITASK_ZIP_MAX_MEMBERS = 16
ITASK_DOWNLOAD_TIMEOUT = 120.0
OSSUTIL_VERSION = "v1.7.19"
OSSUTIL_EXPECTED_VERSION_LINE = "ossutil version: %s" % OSSUTIL_VERSION
OSSUTIL_MANAGED_BIN = os.path.join(ITASK_INSTALL_DIR, "ossutil")
OSSUTIL_INSTALL_LOCK_FILE = os.path.join(ITASK_INSTALL_DIR, ".ossutil-install.lock")
OSSUTIL_DOWNLOAD_MAX_BYTES = 64 * 1024 * 1024
OSSUTIL_BINARY_MAX_BYTES = 32 * 1024 * 1024
OSSUTIL_ZIP_MAX_MEMBERS = 8
OSSUTIL_DOWNLOAD_TIMEOUT = 120.0
OSSUTIL_ARTIFACTS = {
    "darwin-arm64": {
        "label": "macOS arm64",
        "url": "https://gosspublic.alicdn.com/ossutil/current/ossutil-mac-arm64.zip",
        "archive_sha256": "43e71a1516cd7c9cc573c2462b72ce4417f4ddbcd1b2274854b1271beae971ad",
        "member": "ossutil-mac-arm64/ossutil",
        "binary_sha256": "84ca0ada6b040e1814291b78f94159de53a698b5ca5bb007217ba26560b82d4b",
    },
    "linux-amd64": {
        "label": "Linux amd64",
        "url": "https://gosspublic.alicdn.com/ossutil/current/ossutil-linux-amd64.zip",
        "archive_sha256": "f74baa308436e3543b6bc7efb35f8ffe317d670e6797f451acbb4c9f097145d5",
        "member": "ossutil-linux-amd64/ossutil",
        "binary_sha256": "0970061c150262d67e2ff46045ce1237f1cbf0df4fcd9f126802a1dc7e0036bf",
    },
}
SSH_USER = "root"
REMOTE_ROOT = "/root"
LOCAL_MOUNT_REMOTE_ROOT = "/"
DEFAULT_IMAGE = "hcr.meta-wulan01.hw-wulan.local/antsys/vllm:release_0.18.0_0415_202604171255_aarch64"
NAS_MOUNT_RETRIES = 60
NAS_MOUNT_RETRY_INTERVAL = 5.0
NAS_MOUNT_DEEPSEEK = "3374703f-5ad1-4170-8645-4a0a9f932241.sfsturbo.internal:/"
NAS_MOUNT_SFS_TURBO = "2b0eb94f-fcb3-4eaa-9fba-3709b110010e.sfsturbo.internal:/"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
LOCAL_MOUNT_ROOT = os.path.abspath(os.path.expanduser("~/itask-workspace"))
LOCAL_MOUNT_STATE_FILE = os.path.abspath(os.path.expanduser("~/.itask_gui_mounts.json"))
OSS_BUCKET = "antsys-language-compiler"
OSS_PREFIX = "oss://%s" % OSS_BUCKET
OSS_ENDPOINT = "cn-shanghai-alipay-office.oss-alipay.aliyuncs.com"
OSS_CONFIG_FILE = os.path.abspath(os.path.expanduser("~/.ossutilconfig"))
OSS_SYNC_ROOT = os.path.abspath(os.path.expanduser("~"))
# A config file is used only when -f/--config is passed at startup; otherwise
# the app runs on built-in defaults and never reads or writes a config file.
GUI_CONFIG_FILE: Optional[str] = None
EXAMPLE_USER_PLACEHOLDER = "your_employee_id"
DEFAULT_GUI_HOST = "127.0.0.1"
DEFAULT_GUI_PORT = 8765
OSS_STATE_TTL = 3600.0
OSS_ARCHIVE_MAX_MEMBERS = 1000000
OSS_ARCHIVE_MAX_BYTES = 1024 ** 4
OSS_REMOTE_CANDIDATES = (
    "/a3_inference/itask/workdir/shared/ossutil",
    "/sfs_turbo/chenxiao/oss/ossutil",
    "/a2_inference/itask/workdir/shared/ossutil",
)
DEFAULT_OSS_EXCLUDES = ["*.so", "*.abi3.so", "__pycache__", "build", "vllm_flash_attn"]
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
USER_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
IMAGE_PATTERN = re.compile(r"^[^\s]{1,512}$")
DIRECTORY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
NAS_SCRIPT = """#!/bin/bash
set -e
rm -f /etc/yum.repos.d/*.repo
cat > /etc/yum.repos.d/openeuler.repo << 'EOF'
[OS]
name=openEuler-24.03-LTS - OS
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/OS/$basearch/
enabled=1
gpgcheck=0
sslverify=0
[everything]
name=openEuler-24.03-LTS - everything
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/everything/$basearch/
enabled=1
gpgcheck=0
sslverify=0
[EPOL]
name=openEuler-24.03-LTS - EPOL
baseurl=http://mirrors.aliyun.com/openeuler/openEuler-24.03-LTS/EPOL/main/$basearch/
enabled=1
gpgcheck=0
sslverify=0
EOF
yum clean all
yum makecache
mkdir -p /mnt/deepseek /sfs_turbo
yum install -y nfs-utils rsync
mount -t nfs -o vers=3,nolock,noresvport """ + NAS_MOUNT_DEEPSEEK + """ /mnt/deepseek
mount -t nfs -o vers=3,nolock,noresvport """ + NAS_MOUNT_SFS_TURBO + """ /sfs_turbo
"""

app = FastAPI(title="远程文件管理")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def sanitized_validation_error(_: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({key: value for key, value in error.items() if key not in {"input", "ctx", "url"}})
    return JSONResponse(status_code=422, content={"detail": errors})


tunnels: Dict[str, asyncio.subprocess.Process] = {}
tunnel_ports: Dict[str, int] = {}
local_mount_processes: Dict[str, Dict[str, asyncio.subprocess.Process]] = {}
local_mount_paths: Dict[str, Dict[str, str]] = {}
local_mount_created: Dict[str, Dict[str, bool]] = {}
local_mount_ports: Dict[str, Dict[str, int]] = {}
local_mount_preferred_ports: Dict[str, int] = {}
local_mount_watchers: set = set()
mount_states: Dict[str, Dict[str, Any]] = {}
mount_tasks: Dict[str, asyncio.Task] = {}
oss_states: Dict[str, Dict[str, Any]] = {}
oss_tasks: Dict[str, asyncio.Task] = {}
event_loop_locks: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}


def event_loop_lock(name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    current = event_loop_locks.get(name)
    if current is None or current[0] is not loop:
        current = (loop, asyncio.Lock())
        event_loop_locks[name] = current
    return current[1]


def current_oss_task_id() -> Optional[str]:
    for tid, state in oss_states.items():
        if state.get("status") in ("queued", "running"):
            return tid
    return None


def cleanup_old_oss_states() -> None:
    now = time.time()
    for tid in [t for t, s in oss_states.items()
                if s.get("finished_at") and now - s["finished_at"] > OSS_STATE_TTL]:
        oss_states.pop(tid, None)


def validate_name(value: str) -> str:
    if not NAME_PATTERN.fullmatch(value):
        raise HTTPException(400, "invalid pod name")
    return value


def remote_path(path: str) -> str:
    if "\\" in path or "\x00" in path:
        raise HTTPException(400, "invalid file path")
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(REMOTE_ROOT, path or "."))


def _safe_local_resolved(path: str, label: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path))
    try:
        return os.path.realpath(expanded)
    except (OSError, ValueError):
        raise HTTPException(400, "invalid local path")


def safe_local_directory(path: str, must_exist: bool = False) -> str:
    resolved = _safe_local_resolved(path, "本地目录")
    if must_exist and not os.path.isdir(resolved):
        raise HTTPException(400, "本地目录不存在")
    return resolved


def safe_local_path(path: str) -> str:
    return _safe_local_resolved(path, "本地路径")


def safe_remote_directory(path: str) -> str:
    if not path or "\x00" in path or "\\" in path or not path.startswith("/"):
        raise HTTPException(400, "容器目录必须是绝对路径")
    normalized = posixpath.normpath(path)
    if normalized == "." or any(part == ".." for part in path.split("/")):
        raise HTTPException(400, "invalid remote directory")
    return normalized


def safe_oss_key(key: str, allow_empty: bool = False) -> str:
    if not key:
        if allow_empty:
            return ""
        raise HTTPException(400, "invalid OSS object key")
    if (
        not key.strip()
        or key.startswith("/")
        or len(key.encode("utf-8")) > 1024
        or "//" in key
        or any(part == ".." for part in key.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
    ):
        raise HTTPException(400, "invalid OSS object key")
    return key


def safe_oss_prefix(prefix: str) -> str:
    checked = safe_oss_key(prefix, allow_empty=True)
    return "%s/" % checked.rstrip("/") if checked else ""


def automatic_upload_key(filesystem_path: str, source_kind: str, package_upload: bool) -> str:
    source_name = posixpath.basename(filesystem_path.rstrip("/"))
    if not source_name:
        raise HTTPException(400, "不能直接上传文件系统根目录")
    checked_name = safe_oss_key(source_name)
    if source_kind == "file":
        if package_upload:
            raise HTTPException(400, "普通文件不能设置 package_upload")
        return checked_name
    if source_kind != "directory":
        raise HTTPException(400, "无法确定上传来源类型")
    return "%s.tar" % checked_name if package_upload else "%s/" % checked_name


def oss_uri(user: str, key: str = "") -> str:
    if not USER_PATTERN.fullmatch(user):
        raise HTTPException(400, "invalid OSS user")
    checked_key = safe_oss_key(key, allow_empty=True)
    return "%s/%s/%s" % (OSS_PREFIX, user, checked_key) if checked_key else "%s/%s/" % (OSS_PREFIX, user)


def parse_oss_listing(output: str, user: str) -> List[Dict[str, Any]]:
    prefix = "%s/%s/" % (OSS_PREFIX, user)
    objects: List[Dict[str, Any]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("LastModified") or "Object Number" in stripped:
            continue
        idx = line.find(OSS_PREFIX)
        if idx < 0:
            continue
        uri_part = line[idx:]
        key = uri_part[len(prefix):] if uri_part.startswith(prefix) else uri_part[len(OSS_PREFIX) + 1:]
        before = line[:idx].strip()
        parts = before.split()
        size: Optional[int] = None
        modified = ""
        if len(parts) >= 3:
            try:
                size = int(parts[-3])
                timestamp_parts = parts[:-3]
                if (
                    len(timestamp_parts) >= 2
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp_parts[0])
                    and re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", timestamp_parts[1])
                ):
                    modified = "%s %s" % (timestamp_parts[0], timestamp_parts[1][:5])
                else:
                    modified = " ".join(timestamp_parts)
            except ValueError:
                modified = before
        elif parts:
            modified = before
        objects.append({"key": key, "size": size, "modified": modified})
    return objects


def parse_oss_directory_listing(output: str, user: str, prefix: str) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for item in parse_oss_listing(output, user):
        key = item["key"]
        if prefix and not key.startswith(prefix):
            continue
        relative = key[len(prefix):]
        if not relative:
            continue
        first, separator, _ = relative.partition("/")
        if separator:
            directory_key = "%s%s/" % (prefix, first)
            entries[directory_key] = {
                "key": directory_key,
                "name": first,
                "kind": "directory",
                "size": None,
                "modified": "",
            }
        else:
            entries[key] = {
                "key": key,
                "name": relative,
                "kind": "object",
                "size": item["size"],
                "modified": item["modified"],
            }
    return sorted(entries.values(), key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold()))


async def populate_oss_object_metadata(entries: List[Dict[str, Any]], user: str, listing_for_key) -> None:
    semaphore = asyncio.Semaphore(6)

    async def populate(entry: Dict[str, Any]) -> None:
        async with semaphore:
            output = await listing_for_key(entry["key"])
        for item in parse_oss_listing(output, user):
            if item["key"] == entry["key"]:
                entry["size"] = item["size"]
                entry["modified"] = item["modified"]
                return

    await asyncio.gather(*(
        populate(entry)
        for entry in entries
        if entry["kind"] == "object"
        and (entry.get("size") is None or not entry.get("modified"))
    ))


def validate_excludes(excludes: List[str]) -> List[str]:
    if len(excludes) > 32:
        raise HTTPException(400, "排除规则不能超过 32 条")
    result = []
    for pattern in excludes:
        item = pattern.strip()
        if not item:
            continue
        if len(item) > 256 or "\x00" in item or "\n" in item or item.startswith("/") or ".." in item.split("/"):
            raise HTTPException(400, "invalid exclude pattern")
        result.append(item)
    return result


def validate_directory_name(value: str) -> str:
    if not DIRECTORY_NAME_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise HTTPException(400, "invalid directory name")
    return value


def load_gui_config() -> Dict[str, Any]:
    if GUI_CONFIG_FILE is None:
        return {}
    try:
        with open(GUI_CONFIG_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    users = data.get("users")
    if isinstance(users, list):
        # config_example.json ships a placeholder user; never expose it as a
        # real account in the UI or persist it back.
        data["users"] = [user for user in users if user != EXAMPLE_USER_PLACEHOLDER]
    return data


def save_gui_config(config: Dict[str, Any]) -> None:
    if GUI_CONFIG_FILE is None:
        return
    temporary_file = GUI_CONFIG_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_file, GUI_CONFIG_FILE)


def load_gui_settings() -> Dict[str, Any]:
    """Return validated GUI settings from the -f config file (restart to apply)."""
    raw = load_gui_config()
    host = raw.get("host")
    port = raw.get("port")
    auto_mount_nas = raw.get("auto_mount_nas")
    return {
        "host": host.strip() if isinstance(host, str) and host.strip() else DEFAULT_GUI_HOST,
        "port": port if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535 else DEFAULT_GUI_PORT,
        "auto_mount_nas": auto_mount_nas if isinstance(auto_mount_nas, bool) else True,
    }


GUI_SETTINGS = load_gui_settings()
AUTO_MOUNT_NAS = GUI_SETTINGS["auto_mount_nas"]


def parse_gui_args() -> argparse.Namespace:
    """Parse CLI overrides; omitted options keep the config.json settings."""

    def port_number(value: str) -> int:
        port = int(value)
        if not 1 <= port <= 65535:
            raise argparse.ArgumentTypeError("port must be between 1 and 65535")
        return port

    # -h is used for the listen host, so the built-in -h/--help is replaced
    # with a --help-only option.
    parser = argparse.ArgumentParser(description="远程文件管理", add_help=False)
    parser.add_argument(
        "--help",
        action="help",
        help="显示帮助信息并退出",
    )
    parser.add_argument(
        "-f",
        "--config",
        default=None,
        help="配置文件路径（未指定时不使用任何配置文件；指定后创建 Pod 时写回 users/images，"
        "并保存同次指定的 -h/-p）",
    )
    parser.add_argument(
        "-h",
        "--host",
        default=None,
        help="Web 服务监听地址（默认沿用配置文件的 host，未配置时为 %s）" % DEFAULT_GUI_HOST,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=port_number,
        default=None,
        help="Web 服务监听端口（默认沿用配置文件的 port，未配置时为 %d）" % DEFAULT_GUI_PORT,
    )
    parser.add_argument(
        "--auto-mount-nas",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="新 Pod 是否自动挂载 NAS，用 --no-auto-mount-nas 关闭"
        "（默认沿用配置文件的 auto_mount_nas，未配置时为开启）",
    )
    return parser.parse_args()


def env() -> Dict[str, str]:
    result = {"PATH": os.getenv("PATH", ""), "HOME": os.getenv("HOME", "")}
    for key in ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "LANG", "LC_ALL"):
        if os.getenv(key):
            result[key] = os.getenv(key)
    return result

async def run(args: List[str], timeout: float = 120) -> Dict[str, str]:
    try:
        process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env())
    except FileNotFoundError:
        raise HTTPException(501, "command not found: %s" % args[0])
    except OSError as exc:
        raise HTTPException(502, "无法执行 %s：%s" % (args[0], exc))
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill(); await process.wait(); raise HTTPException(504, "command timed out")
    except asyncio.CancelledError:
        process.kill(); await process.wait(); raise
    output, error = stdout.decode(errors="replace").strip(), stderr.decode(errors="replace").strip()
    if process.returncode:
        raise HTTPException(502, error or output or "command failed")
    return {"output": output, "error": error}


def ssh_control_path(name: str, port: int) -> str:
    # The control socket lives in a short private directory: unix socket paths
    # are limited to ~104 bytes, and /tmp is shared between users.
    base = "/tmp/itask-gui-%d" % os.getuid()
    os.makedirs(base, mode=0o700, exist_ok=True)
    return os.path.join(base, "ssh-%s-%d.sock" % (name, port))


def ssh_args(name: str, port: int) -> List[str]:
    # ControlMaster/ControlPersist reuse one authenticated connection per pod
    # tunnel, so repeated commands (directory listings, OSS operations) skip
    # the full SSH handshake. The master never expires on its own: it is bound
    # to the tunnel connection and stop_tunnel shuts it down explicitly.
    # ServerAlive keeps the idle connection warm through NATs and lets a dead
    # master exit, after which a fresh one is created automatically.
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ControlMaster=auto", "-o", "ControlPath=" + ssh_control_path(name, port), "-o", "ControlPersist=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-p", str(port), "%s@127.0.0.1" % SSH_USER]


class CreatePod(BaseModel):
    name: str
    image: str = DEFAULT_IMAGE
    user: str = ""

    @validator("name")
    def name_ok(cls, value): return validate_name(value)
    @validator("image")
    def image_ok(cls, value):
        if not IMAGE_PATTERN.fullmatch(value): raise ValueError("invalid image")
        return value
    @validator("user", always=True)
    def user_ok(cls, value):
        if not USER_PATTERN.fullmatch(value): raise ValueError("invalid user")
        return value


class OssConfigRequest(BaseModel):
    access_key_id: str
    access_key_secret: SecretStr
    replace_existing: bool = False
    model_config = {"extra": "forbid"}

    @validator("access_key_id")
    def access_key_id_ok(cls, value):
        value = value.strip()
        if not value or len(value) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("invalid AccessKey ID")
        return value

    @validator("access_key_secret")
    def access_key_secret_ok(cls, value):
        secret = value.get_secret_value()
        if not secret or len(secret) > 512 or secret != secret.strip() or any(ord(char) < 32 or ord(char) == 127 for char in secret):
            raise ValueError("invalid AccessKey Secret")
        return value


class OssUnifiedTransferRequest(BaseModel):
    user: str
    direction: str
    filesystem_path: str
    oss_key: str = ""
    source_kind: str = "auto"
    package_upload: bool = False
    extract_download: bool = False
    conflict_policy: str = "overwrite"
    excludes: List[str] = DEFAULT_OSS_EXCLUDES
    model_config = {"extra": "forbid"}

    @validator("user")
    def user_ok(cls, value):
        if not USER_PATTERN.fullmatch(value): raise ValueError("invalid OSS user")
        return value
    @validator("direction")
    def direction_ok(cls, value):
        if value not in {"upload", "download"}: raise ValueError("invalid direction")
        return value
    @validator("filesystem_path")
    def filesystem_path_ok(cls, value):
        if not value.strip(): raise ValueError("filesystem_path required")
        return value
    @validator("oss_key")
    def oss_key_ok(cls, value): return safe_oss_key(value, allow_empty=True)
    @validator("source_kind")
    def source_kind_ok(cls, value):
        if value not in {"auto", "file", "directory", "object", "prefix"}: raise ValueError("invalid source kind")
        return value
    @validator("conflict_policy")
    def conflict_policy_ok(cls, value):
        if value not in {"skip", "overwrite"}: raise ValueError("invalid conflict policy")
        return value
    @validator("excludes")
    def excludes_ok(cls, value): return validate_excludes(value)


class CreateDirectoryRequest(BaseModel):
    path: str
    name: str

    @validator("name")
    def directory_name_ok(cls, value): return validate_directory_name(value)


class RenameRequest(BaseModel):
    path: str
    name: str

    @validator("name")
    def rename_name_ok(cls, value): return validate_directory_name(value)


def parse_pods(output: str) -> List[Dict[str, str]]:
    result = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"no\s+tasks?\s+found[.!]?", stripped, re.IGNORECASE):
            continue
        parts = stripped.split()
        if parts and NAME_PATTERN.fullmatch(parts[0]) and parts[0].lower() not in {"name", "pod", "id"}:
            result.append({"name": parts[0], "status": " ".join(parts[1:]) or "unknown"})
    return result


def parse_pods_json(output: str) -> List[Dict[str, Any]]:
    data = json.loads(output)
    if not isinstance(data, list):
        raise ValueError("itask list JSON is not an array")
    result: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not NAME_PATTERN.fullmatch(name):
            continue
        status = str(item.get("status") or item.get("pod_status") or "unknown")
        entry: Dict[str, Any] = {
            "name": name,
            "status": status,
            "pod_status": str(item.get("pod_status") or status),
            "node_status": str(item.get("node_status") or ""),
            "user": str(item.get("user") or ""),
            "workspace_name": str(item.get("workspace_name") or ""),
            "created_at": str(item.get("created_at") or ""),
        }
        resources = item.get("resources")
        if isinstance(resources, dict):
            entry["resources"] = {
                key: str(resources.get(key) or "")
                for key in ("cpu", "memory", "gpu", "gpu_type")
            }
        result.append(entry)
    return result


async def list_itask_pods() -> Dict[str, Any]:
    executable = require_itask_bin()
    try:
        result = await run([executable, "list", "--output", "json"])
        return {"output": result["output"], "pods": parse_pods_json(result["output"])}
    except (HTTPException, ValueError, json.JSONDecodeError):
        result = await run([executable, "list"])
        return {"output": result["output"], "pods": parse_pods(result["output"])}


def parse_file_tree_entries(output: str, show_hidden: bool) -> List[Dict[str, Any]]:
    """Aggregate a two-level find listing into one-level entries.

    Lines without "/" are the listed directory's entries; "parent/child" lines
    only prove that parent is a non-empty directory.
    """
    entries: List[Dict[str, Any]] = []
    directories: Dict[str, Dict[str, Any]] = {}
    lines = output.splitlines()
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 3 and "/" not in parts[0]:
            if not show_hidden and parts[0].startswith("."):
                continue
            entry: Dict[str, Any] = {"name": parts[0], "kind": parts[1], "size": parts[2]}
            if entry["kind"] == "d":
                entry["empty"] = True
                directories[entry["name"]] = entry
            entries.append(entry)
    for line in lines:
        target = line.split("\t", 1)[0]
        if "/" in target:
            parent = directories.get(target.split("/", 1)[0])
            if parent is not None:
                parent["empty"] = False
    return entries


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


TUNNEL_READY_TIMEOUT = 30.0
TUNNEL_READY_INTERVAL = 0.5


async def wait_tunnel_ready(port: int, timeout: float = TUNNEL_READY_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError):
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
            writer.close()
            await writer.wait_closed()
            return
        await asyncio.sleep(TUNNEL_READY_INTERVAL)
    raise HTTPException(502, "隧道端口 %d 未就绪，请稍后重试" % port)


async def check_tunnel_ssh(name: str, port: int) -> bool:
    try:
        await run(ssh_args(name, port) + ["-o", "ConnectTimeout=5", "true"], 10)
        return True
    except HTTPException:
        return False


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_mount_path(name: str, remote_path: str = "/") -> str:
    validate_name(name)
    sub = "root_fs" if remote_path in ("", "/") else remote_path.strip("/").replace("/", "_")
    return os.path.join(LOCAL_MOUNT_ROOT, name, sub)


def normalize_local_mount_path(path: str) -> str:
    if not path or "\x00" in path:
        raise HTTPException(400, "本地挂载目录不能为空")
    normalized = os.path.abspath(os.path.expanduser(path.strip()))
    if normalized == "/":
        raise HTTPException(400, "不能将文件系统根目录用作挂载目录")
    if os.path.islink(normalized):
        raise HTTPException(400, "本地挂载目录不能是符号链接")
    return normalized


def save_local_mount_state() -> bool:
    records = []
    for name, paths in local_mount_paths.items():
        for remote_path, local_path in paths.items():
            records.append({
                "pod": name,
                "remote_path": remote_path,
                "local_path": local_path,
                "created_by_app": bool(local_mount_created.get(name, {}).get(remote_path)),
                "port": local_mount_ports.get(name, {}).get(remote_path),
            })
    temporary_path = ""
    try:
        parent = os.path.dirname(LOCAL_MOUNT_STATE_FILE) or "."
        os.makedirs(parent, mode=0o700, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".itask_gui_mounts.", dir=parent, text=True)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(
                {
                    "version": 1,
                    "mounts": records,
                    "preferred_ports": {
                        name: tunnel_ports.get(name, local_mount_preferred_ports.get(name))
                        for name in set(local_mount_preferred_ports) | set(tunnel_ports)
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, LOCAL_MOUNT_STATE_FILE)
        os.chmod(LOCAL_MOUNT_STATE_FILE, 0o600)
        return True
    except OSError:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)
        return False


def load_local_mount_state() -> None:
    try:
        with open(LOCAL_MOUNT_STATE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return
    records = data.get("mounts", []) if isinstance(data, dict) else []
    if not isinstance(records, list):
        return
    preferred_ports = data.get("preferred_ports", {})
    if isinstance(preferred_ports, dict):
        for raw_name, raw_port in preferred_ports.items():
            try:
                name = validate_name(str(raw_name))
                port = int(raw_port)
            except (HTTPException, TypeError, ValueError):
                continue
            if 0 < port < 65536:
                local_mount_preferred_ports[name] = port
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            name = validate_name(str(record.get("pod", "")))
            remote_path = safe_remote_directory(str(record.get("remote_path", "")))
            local_path = normalize_local_mount_path(str(record.get("local_path", "")))
            port = int(record.get("port") or 0)
        except (HTTPException, TypeError, ValueError):
            continue
        local_mount_paths.setdefault(name, {})[remote_path] = local_path
        local_mount_created.setdefault(name, {})[remote_path] = record.get("created_by_app") is True
        if 0 < port < 65536:
            local_mount_ports.setdefault(name, {})[remote_path] = port


def discover_existing_local_mounts() -> None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "command="],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    commands = [line.strip() for line in result.stdout.decode(errors="replace").splitlines()]
    tunnel_pattern = re.compile(
        r"^(?:\S*/)?itask\s+ssh-tunnel\s+([A-Za-z0-9._-]+)\s+--port\s+(\d+)(?:\s|$)"
    )
    sshfs_pattern = re.compile(
        r"^(?:\S*/)?sshfs(?:\s+-f)?\s+%s@127\.0\.0\.1:(\S+)\s+(.+?)\s+-p\s+(\d+)(?:\s|$)"
        % re.escape(SSH_USER)
    )
    legacy_volume_pattern = re.compile(
        r"(?:^|\s)-o\s+volname=itask-([A-Za-z0-9._-]+)(?:\s|$)"
    )
    port_to_name: Dict[int, str] = {}
    changed = False
    for command in commands:
        match = tunnel_pattern.match(command)
        if not match:
            continue
        try:
            name = validate_name(match.group(1))
            port = int(match.group(2))
        except (HTTPException, TypeError, ValueError):
            continue
        if 0 < port < 65536:
            port_to_name[port] = name
            local_mount_preferred_ports[name] = port
            changed = True

    for command in commands:
        match = sshfs_pattern.match(command)
        if not match:
            continue
        try:
            remote_path = safe_remote_directory(match.group(1))
            local_path = normalize_local_mount_path(match.group(2))
            port = int(match.group(3))
            name = port_to_name.get(port)
            if name is None:
                legacy_match = legacy_volume_pattern.search(command)
                if legacy_match:
                    name = validate_name(legacy_match.group(1))
            if name is None:
                continue
        except (HTTPException, TypeError, ValueError):
            continue
        if not os.path.ismount(local_path) or remote_path in local_mount_paths.get(name, {}):
            continue
        local_mount_paths.setdefault(name, {})[remote_path] = local_path
        local_mount_created.setdefault(name, {})[remote_path] = False
        local_mount_ports.setdefault(name, {})[remote_path] = port
        changed = True
    if changed:
        save_local_mount_state()


load_local_mount_state()
discover_existing_local_mounts()


def local_mount_status(name: str) -> Dict[str, Any]:
    mounts = []
    for remote_path, local_path in local_mount_paths.get(name, {}).items():
        process = local_mount_processes.get(name, {}).get(remote_path)
        mounts.append({
            "remote_path": remote_path,
            "local_path": local_path,
            "mounted": os.path.ismount(local_path),
            "process_alive": process is not None and process.returncode is None,
        })
    mounts.sort(key=lambda item: (item["remote_path"], item["local_path"]))
    return {"mounts": mounts}


def active_local_mount_port(name: str) -> Optional[int]:
    preferred_port = local_mount_preferred_ports.get(name)
    if isinstance(preferred_port, int) and 0 < preferred_port < 65536:
        for remote_path, local_path in local_mount_paths.get(name, {}).items():
            if local_mount_ports.get(name, {}).get(remote_path) == preferred_port and os.path.ismount(local_path):
                return preferred_port
    for remote_path, local_path in local_mount_paths.get(name, {}).items():
        port = local_mount_ports.get(name, {}).get(remote_path)
        if isinstance(port, int) and 0 < port < 65536 and os.path.ismount(local_path):
            return port
    return None


async def unmount_local_path(path: str, force: bool = False) -> None:
    if not os.path.ismount(path):
        return
    cmd = ["diskutil", "unmount", "force" if force else "", path]
    cmd = [c for c in cmd if c]
    try:
        await run(cmd, 30)
    except HTTPException as exc:
        if not force:
            raise HTTPException(502, "卸载失败，请关闭正在使用该目录的程序后重试，或使用强制卸载：%s" % exc.detail)
        raise


async def terminate_local_mount_process(process: Optional[asyncio.subprocess.Process]) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def watch_local_mount_process(name: str, remote_path: str, process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(Exception):
        await process.communicate()
    if local_mount_processes.get(name, {}).get(remote_path) is process:
        local_mount_processes.get(name, {}).pop(remote_path, None)


def track_local_mount_process(name: str, remote_path: str, process: asyncio.subprocess.Process) -> None:
    task = asyncio.create_task(watch_local_mount_process(name, remote_path, process))
    local_mount_watchers.add(task)
    task.add_done_callback(local_mount_watchers.discard)


def local_mount_volume_name(name: str, local_path: str) -> str:
    volume_name = os.path.basename(local_path.rstrip(os.sep)).strip()
    if not volume_name:
        volume_name = "itask-%s" % name
    return volume_name.replace(",", "_")[:128]


async def start_local_mount(name: str, local_path: str, remote_path: str, port: int) -> asyncio.subprocess.Process:
    sshfs = shutil.which("sshfs")
    if not sshfs:
        raise HTTPException(501, "未检测到 sshfs")
    process = await asyncio.create_subprocess_exec(
        sshfs, "-f", "%s@127.0.0.1:%s" % (SSH_USER, remote_path), local_path, "-p", str(port),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "reconnect",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "defer_permissions",
        "-o", "cache=yes", "-o", "kernel_cache",
        "-o", "volname=%s" % local_mount_volume_name(name, local_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env(),
    )
    for _ in range(150):
        if os.path.ismount(local_path):
            return process
        if process.returncode is not None:
            _, stderr = await process.communicate()
            raise HTTPException(502, stderr.decode(errors="replace").strip() or "sshfs exited before mounting")
        await asyncio.sleep(.1)
    await terminate_local_mount_process(process)
    if os.path.ismount(local_path):
        with contextlib.suppress(HTTPException):
            await unmount_local_path(local_path, True)
    raise HTTPException(504, "等待 SSHFS 挂载超时")


async def unmount_local_record(name: str, remote_path: str, force: bool = False) -> None:
    local_path = local_mount_paths.get(name, {}).get(remote_path)
    if local_path is None:
        return
    await unmount_local_path(local_path, force)
    process = local_mount_processes.get(name, {}).pop(remote_path, None)
    await terminate_local_mount_process(process)
    created_by_app = bool(local_mount_created.get(name, {}).get(remote_path))
    local_mount_paths.get(name, {}).pop(remote_path, None)
    local_mount_created.get(name, {}).pop(remote_path, None)
    local_mount_ports.get(name, {}).pop(remote_path, None)
    if created_by_app:
        with contextlib.suppress(OSError):
            os.rmdir(local_path)
    for registry in (local_mount_processes, local_mount_paths, local_mount_created, local_mount_ports):
        if not registry.get(name):
            registry.pop(name, None)
    save_local_mount_state()


def inspect_oss_config() -> Dict[str, Any]:
    status_data = {"exists": False, "valid": False, "detail": "未配置"}
    if not os.path.lexists(OSS_CONFIG_FILE):
        return status_data
    status_data["exists"] = True
    try:
        file_stat = os.stat(OSS_CONFIG_FILE)
        if not stat.S_ISREG(file_stat.st_mode):
            status_data["detail"] = "OSS 配置路径不是普通文件"
            return status_data
        parser = configparser.RawConfigParser(interpolation=None)
        with open(OSS_CONFIG_FILE, "r", encoding="utf-8") as handle:
            parser.read_file(handle)
        credentials = parser["Credentials"]
        status_data["valid"] = (
            credentials.get("endpoint", "").strip() == OSS_ENDPOINT
            and bool(credentials.get("accessKeyID", "").strip())
            and bool(credentials.get("accessKeySecret", "").strip())
        )
        status_data["detail"] = "已配置" if status_data["valid"] else "配置内容无效"
    except (OSError, UnicodeError, configparser.Error, KeyError):
        status_data["detail"] = "配置内容无效或无法读取"
    return status_data


def _is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_local_gui_request(request: Request, marker: str, detail: str) -> None:
    client_host = request.client.host if request.client else None
    origin = request.headers.get("origin", "")
    parsed_origin = urlparse(origin) if origin else None
    origin_allowed = (
        not origin
        or (
            parsed_origin is not None
            and parsed_origin.scheme in {"http", "https"}
            and _is_loopback_host(parsed_origin.hostname)
        )
    )
    if (
        request.headers.get("x-itask-gui-request") != marker
        or not _is_loopback_host(client_host)
        or not _is_loopback_host(request.url.hostname)
        or not origin_allowed
    ):
        raise HTTPException(403, detail)


def require_local_credentials_request(request: Request) -> None:
    require_local_gui_request(request, "oss-config-dialog", "仅允许从本机 GUI 读取 OSS 配置")


def require_local_itask_install_request(request: Request) -> None:
    require_local_gui_request(request, "itask-install", "仅允许从本机 GUI 安装 itask")


def require_local_ossutil_install_request(request: Request) -> None:
    require_local_gui_request(request, "ossutil-install", "仅允许从本机 GUI 安装 ossutil")


def itask_platform_info(system: Optional[str] = None, machine: Optional[str] = None) -> Dict[str, Any]:
    system_name = (system or platform.system()).strip().lower()
    machine_name = (machine or platform.machine()).strip().lower()
    if system_name == "darwin" and machine_name in {"arm64", "aarch64"}:
        return {"supported": True, "platform": "darwin-arm64", "label": "macOS arm64", "artifact": "itask-darwin-arm64"}
    if system_name == "linux" and machine_name in {"x86_64", "amd64"}:
        return {"supported": True, "platform": "linux-amd64", "label": "Linux amd64", "artifact": "itask-linux-amd64"}
    label = "%s %s" % (system or platform.system(), machine or platform.machine())
    return {"supported": False, "platform": "%s-%s" % (system_name, machine_name), "label": label.strip(), "artifact": None}


def _is_executable_file(path: Optional[str]) -> bool:
    return bool(path and os.path.isfile(path) and os.access(path, os.X_OK))


def ossutil_platform_info(system: Optional[str] = None, machine: Optional[str] = None) -> Dict[str, Any]:
    system_name = (system or platform.system()).strip().lower()
    machine_name = (machine or platform.machine()).strip().lower()
    if system_name == "darwin" and machine_name in {"arm64", "aarch64"}:
        platform_key = "darwin-arm64"
    elif system_name == "linux" and machine_name in {"x86_64", "amd64"}:
        platform_key = "linux-amd64"
    else:
        platform_key = "%s-%s" % (system_name, machine_name)
    artifact = OSSUTIL_ARTIFACTS.get(platform_key)
    label = artifact["label"] if artifact else ("%s %s" % (system or platform.system(), machine or platform.machine())).strip()
    return {
        "supported": artifact is not None,
        "platform": platform_key,
        "label": label,
        "artifact": artifact,
    }


def resolve_ossutil_bin() -> Optional[str]:
    system_binary = shutil.which("ossutil")
    if _is_executable_file(system_binary):
        return os.path.abspath(system_binary)
    if _is_executable_file(OSSUTIL_MANAGED_BIN):
        return OSSUTIL_MANAGED_BIN
    return None


def resolve_itask_bin() -> Optional[str]:
    system_binary = shutil.which("itask")
    if _is_executable_file(system_binary):
        return os.path.abspath(system_binary)
    if _is_executable_file(ITASK_MANAGED_BIN):
        return ITASK_MANAGED_BIN
    return None


def require_itask_bin() -> str:
    executable = resolve_itask_bin()
    if not executable:
        raise HTTPException(501, "未检测到 itask，请在页面点击安装")
    return executable


def display_home_path(path: str) -> str:
    home = os.path.abspath(os.path.expanduser("~"))
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, home)
    except ValueError:
        return absolute
    if relative == ".":
        return "~"
    if relative != ".." and not relative.startswith(".." + os.sep):
        return "~/" + relative.replace(os.sep, "/")
    return absolute


def validate_itask_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return
    raise HTTPException(502, "itask 下载地址必须使用 HTTPS")


def prepare_itask_candidate(artifact: str) -> str:
    validate_itask_download_url(DEFAULT_ITASK_DOWNLOAD_URL)
    archive_path = ""
    candidate_dir = ""
    candidate_path = ""
    prepared = False
    try:
        os.makedirs(ITASK_INSTALL_DIR, mode=0o700, exist_ok=True)
        archive_fd, archive_path = tempfile.mkstemp(prefix=".itask-download-", suffix=".zip", dir=ITASK_INSTALL_DIR)
        request = urllib_request.Request(DEFAULT_ITASK_DOWNLOAD_URL, headers={"User-Agent": "itask-gui/1.0"})
        with os.fdopen(archive_fd, "wb") as output, urllib_request.urlopen(request, timeout=ITASK_DOWNLOAD_TIMEOUT) as response:
            validate_itask_download_url(response.geturl())
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > ITASK_DOWNLOAD_MAX_BYTES:
                raise HTTPException(502, "itask 下载文件超过大小限制")
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > ITASK_DOWNLOAD_MAX_BYTES:
                    raise HTTPException(502, "itask 下载文件超过大小限制")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not zipfile.is_zipfile(archive_path):
            raise HTTPException(502, "itask 下载内容不是有效 ZIP 文件")

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            if len(members) > ITASK_ZIP_MAX_MEMBERS:
                raise HTTPException(502, "itask ZIP 文件成员过多")
            selected = []
            for member in members:
                name = member.filename
                normalized = posixpath.normpath(name)
                parts = [part for part in name.split("/") if part not in {"", "."}]
                if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part == ".." for part in parts) or normalized.startswith("../"):
                    raise HTTPException(502, "itask ZIP 包含不安全路径")
                if posixpath.basename(normalized) == artifact:
                    selected.append(member)
            if len(selected) != 1:
                raise HTTPException(502, "itask ZIP 中未找到唯一的 %s" % artifact)
            member = selected[0]
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if member.is_dir() or file_type == stat.S_IFLNK or file_type not in {0, stat.S_IFREG}:
                raise HTTPException(502, "itask ZIP 中的目标不是普通文件")
            if member.file_size <= 0 or member.file_size > ITASK_BINARY_MAX_BYTES:
                raise HTTPException(502, "itask 二进制大小无效")
            if member.compress_size > 0 and member.file_size > member.compress_size * 200:
                raise HTTPException(502, "itask ZIP 压缩比异常")

            candidate_dir = tempfile.mkdtemp(prefix=".itask-candidate-", dir=ITASK_INSTALL_DIR)
            candidate_path = os.path.join(candidate_dir, "itask")
            candidate_fd = os.open(
                candidate_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o700,
            )
            digest = hashlib.sha256()
            with os.fdopen(candidate_fd, "wb") as output, archive.open(member, "r") as source:
                os.fchmod(output.fileno(), 0o755)
                extracted = 0
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    extracted += len(block)
                    if extracted > ITASK_BINARY_MAX_BYTES:
                        raise HTTPException(502, "itask 二进制超过大小限制")
                    digest.update(block)
                    output.write(block)
                if extracted != member.file_size:
                    raise HTTPException(502, "itask 二进制解压不完整")
                output.flush()
                os.fsync(output.fileno())
            expected_digest = ITASK_ARTIFACT_SHA256.get(artifact)
            if expected_digest and digest.hexdigest() != expected_digest:
                raise HTTPException(502, "itask 二进制 SHA-256 校验失败")
        prepared = True
        return candidate_path
    except HTTPException:
        raise
    except urllib_error.HTTPError as exc:
        raise HTTPException(502, "下载 itask 失败：HTTP %d" % exc.code)
    except (urllib_error.URLError, TimeoutError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise HTTPException(502, "下载或解压 itask 失败：%s" % str(exc)[:240])
    finally:
        if archive_path:
            with contextlib.suppress(OSError):
                os.unlink(archive_path)
        if candidate_path and not prepared:
            with contextlib.suppress(OSError):
                os.unlink(candidate_path)
        if candidate_dir and not prepared:
            with contextlib.suppress(OSError):
                os.rmdir(candidate_dir)


def cleanup_itask_candidate(candidate_path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(candidate_path)
    candidate_dir = os.path.dirname(candidate_path)
    if os.path.dirname(candidate_dir) == ITASK_INSTALL_DIR and os.path.basename(candidate_dir).startswith(".itask-candidate-"):
        with contextlib.suppress(OSError):
            os.rmdir(candidate_dir)


def publish_itask_candidate(candidate_path: str) -> int:
    candidate_inode = os.stat(candidate_path, follow_symlinks=False).st_ino
    candidate_dir = os.path.dirname(candidate_path)
    os.replace(candidate_path, ITASK_MANAGED_BIN)
    if os.path.dirname(candidate_dir) == ITASK_INSTALL_DIR and os.path.basename(candidate_dir).startswith(".itask-candidate-"):
        with contextlib.suppress(OSError):
            os.rmdir(candidate_dir)
    with contextlib.suppress(OSError):
        directory_fd = os.open(ITASK_INSTALL_DIR, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return candidate_inode


def remove_published_itask(expected_inode: int) -> None:
    try:
        current = os.stat(ITASK_MANAGED_BIN, follow_symlinks=False)
        if current.st_ino != expected_inode:
            return
        os.unlink(ITASK_MANAGED_BIN)
    except FileNotFoundError:
        return
    with contextlib.suppress(OSError):
        directory_fd = os.open(ITASK_INSTALL_DIR, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def ensure_ossutil_install_dir() -> None:
    try:
        os.makedirs(ITASK_INSTALL_DIR, mode=0o700, exist_ok=True)
        directory_stat = os.lstat(ITASK_INSTALL_DIR)
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            raise HTTPException(409, "ossutil 托管路径不是安全目录")
        if directory_stat.st_uid != os.geteuid():
            raise HTTPException(409, "ossutil 托管目录不属于当前用户")
        os.chmod(ITASK_INSTALL_DIR, 0o700)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(502, "无法准备 ossutil 托管目录：%s" % exc)


def acquire_ossutil_install_lock() -> int:
    ensure_ossutil_install_dir()
    descriptor = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(OSSUTIL_INSTALL_LOCK_FILE, flags, 0o600)
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid():
            raise HTTPException(409, "ossutil 安装锁文件不安全")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except Exception:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        raise


def release_ossutil_install_lock(descriptor: int) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(descriptor)


class RejectOssutilRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def prepare_ossutil_candidate(platform_info: Dict[str, Any]) -> str:
    artifact = platform_info.get("artifact")
    if not isinstance(artifact, dict):
        raise HTTPException(409, "当前平台不支持自动安装 ossutil")
    url = str(artifact["url"])
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "gosspublic.alicdn.com":
        raise HTTPException(502, "ossutil 制品地址不安全")
    ensure_ossutil_install_dir()
    archive_path = ""
    candidate_dir = ""
    candidate_path = ""
    prepared = False
    try:
        archive_fd, archive_path = tempfile.mkstemp(prefix=".ossutil-download-", suffix=".zip", dir=ITASK_INSTALL_DIR)
        archive_digest = hashlib.sha256()
        request = urllib_request.Request(url, headers={"User-Agent": "itask-gui/1.0"})
        opener = urllib_request.build_opener(RejectOssutilRedirectHandler())
        with os.fdopen(archive_fd, "wb") as output, opener.open(request, timeout=OSSUTIL_DOWNLOAD_TIMEOUT) as response:
            if response.geturl() != url:
                raise HTTPException(502, "ossutil 下载发生了未允许的重定向")
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > OSSUTIL_DOWNLOAD_MAX_BYTES:
                raise HTTPException(502, "ossutil 下载文件超过大小限制")
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > OSSUTIL_DOWNLOAD_MAX_BYTES:
                    raise HTTPException(502, "ossutil 下载文件超过大小限制")
                archive_digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if archive_digest.hexdigest() != artifact["archive_sha256"]:
            raise HTTPException(502, "ossutil ZIP SHA-256 校验失败；官方 current 制品可能已更新，请更新 GUI")
        if not zipfile.is_zipfile(archive_path):
            raise HTTPException(502, "ossutil 下载内容不是有效 ZIP 文件")

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            if len(members) > OSSUTIL_ZIP_MAX_MEMBERS:
                raise HTTPException(502, "ossutil ZIP 文件成员过多")
            selected = []
            for member in members:
                name = member.filename
                normalized = posixpath.normpath(name)
                parts = [part for part in name.split("/") if part not in {"", "."}]
                if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part == ".." for part in parts) or normalized.startswith("../"):
                    raise HTTPException(502, "ossutil ZIP 包含不安全路径")
                mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise HTTPException(502, "ossutil ZIP 包含非普通文件")
                if normalized == artifact["member"]:
                    selected.append(member)
            if len(selected) != 1:
                raise HTTPException(502, "ossutil ZIP 中未找到唯一的目标二进制")
            member = selected[0]
            mode = (member.external_attr >> 16) & 0xFFFF
            if member.is_dir() or stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise HTTPException(502, "ossutil ZIP 中的目标不是普通文件")
            if member.file_size <= 0 or member.file_size > OSSUTIL_BINARY_MAX_BYTES:
                raise HTTPException(502, "ossutil 二进制大小无效")
            if member.compress_size > 0 and member.file_size > member.compress_size * 100:
                raise HTTPException(502, "ossutil ZIP 压缩比异常")

            candidate_dir = tempfile.mkdtemp(prefix=".ossutil-candidate-", dir=ITASK_INSTALL_DIR)
            candidate_path = os.path.join(candidate_dir, "ossutil")
            candidate_fd = os.open(
                candidate_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o700,
            )
            binary_digest = hashlib.sha256()
            with os.fdopen(candidate_fd, "wb") as output, archive.open(member, "r") as source:
                os.fchmod(output.fileno(), 0o755)
                extracted = 0
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    extracted += len(block)
                    if extracted > OSSUTIL_BINARY_MAX_BYTES:
                        raise HTTPException(502, "ossutil 二进制超过大小限制")
                    binary_digest.update(block)
                    output.write(block)
                if extracted != member.file_size:
                    raise HTTPException(502, "ossutil 二进制解压不完整")
                output.flush()
                os.fsync(output.fileno())
            if binary_digest.hexdigest() != artifact["binary_sha256"]:
                raise HTTPException(502, "ossutil 二进制 SHA-256 校验失败")
        prepared = True
        return candidate_path
    except HTTPException:
        raise
    except urllib_error.HTTPError as exc:
        raise HTTPException(502, "下载 ossutil 失败：HTTP %d" % exc.code)
    except (urllib_error.URLError, TimeoutError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise HTTPException(502, "下载或解压 ossutil 失败：%s" % str(exc)[:240])
    finally:
        if archive_path:
            with contextlib.suppress(OSError):
                os.unlink(archive_path)
        if candidate_path and not prepared:
            with contextlib.suppress(OSError):
                os.unlink(candidate_path)
        if candidate_dir and not prepared:
            with contextlib.suppress(OSError):
                os.rmdir(candidate_dir)


def cleanup_ossutil_candidate(candidate_path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(candidate_path)
    candidate_dir = os.path.dirname(candidate_path)
    if os.path.dirname(candidate_dir) == ITASK_INSTALL_DIR and os.path.basename(candidate_dir).startswith(".ossutil-candidate-"):
        with contextlib.suppress(OSError):
            os.rmdir(candidate_dir)


def publish_ossutil_candidate(candidate_path: str) -> int:
    candidate_inode = os.stat(candidate_path, follow_symlinks=False).st_ino
    candidate_dir = os.path.dirname(candidate_path)
    os.replace(candidate_path, OSSUTIL_MANAGED_BIN)
    if os.path.dirname(candidate_dir) == ITASK_INSTALL_DIR and os.path.basename(candidate_dir).startswith(".ossutil-candidate-"):
        with contextlib.suppress(OSError):
            os.rmdir(candidate_dir)
    return candidate_inode


def fsync_ossutil_install_dir() -> None:
    directory_fd = os.open(ITASK_INSTALL_DIR, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


async def await_install_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_requested = True
    return task.result(), cancellation_requested


def remove_published_ossutil(expected_inode: int) -> None:
    try:
        current = os.stat(OSSUTIL_MANAGED_BIN, follow_symlinks=False)
        if current.st_ino != expected_inode:
            return
        os.unlink(OSSUTIL_MANAGED_BIN)
    except FileNotFoundError:
        return
    with contextlib.suppress(OSError):
        directory_fd = os.open(ITASK_INSTALL_DIR, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def read_oss_credentials() -> Dict[str, str]:
    if not os.path.lexists(OSS_CONFIG_FILE):
        raise HTTPException(404, "OSS 配置不存在")
    target_path = os.path.realpath(OSS_CONFIG_FILE)
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target_path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HTTPException(409, "OSS 配置路径不是普通文件")
        parser = configparser.RawConfigParser(interpolation=None)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            parser.read_file(handle)
        credentials = parser["Credentials"]
        access_key_id = credentials.get("accessKeyID", "").strip()
        access_key_secret = credentials.get("accessKeySecret", "").strip()
        if not access_key_id or not access_key_secret:
            raise HTTPException(409, "OSS 配置缺少凭据")
        return {"access_key_id": access_key_id, "access_key_secret": access_key_secret}
    except HTTPException:
        raise
    except (OSError, UnicodeError, configparser.Error, KeyError):
        raise HTTPException(409, "OSS 配置内容无效或无法读取")
    finally:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def require_oss_config() -> None:
    config_status = inspect_oss_config()
    if not config_status["valid"]:
        raise HTTPException(501, "OSS 配置不可用：%s" % config_status["detail"])


def write_oss_config(access_key_id: str, access_key_secret: str, replace_existing: bool) -> bool:
    existed = os.path.lexists(OSS_CONFIG_FILE)
    target_path = os.path.realpath(OSS_CONFIG_FILE) if existed else OSS_CONFIG_FILE
    if existed:
        if not replace_existing:
            raise HTTPException(409, "OSS 配置已存在，请明确选择重新配置")
        if os.path.exists(OSS_CONFIG_FILE):
            try:
                if not stat.S_ISREG(os.stat(OSS_CONFIG_FILE).st_mode):
                    raise HTTPException(409, "现有 OSS 配置路径不是普通文件，无法覆盖")
            except OSError:
                raise HTTPException(500, "无法检查 OSS 配置文件")

    parent = os.path.dirname(target_path)
    if not os.path.isdir(parent):
        raise HTTPException(500, "OSS 配置目录不存在")
    temporary_path = ""
    try:
        descriptor, temporary_path = tempfile.mkstemp(prefix=".ossutilconfig.", dir=parent, text=True)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write("[Credentials]\n")
            handle.write("language=CH\n")
            handle.write("endpoint=%s\n" % OSS_ENDPOINT)
            handle.write("accessKeyID=%s\n" % access_key_id)
            handle.write("accessKeySecret=%s\n" % access_key_secret)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_path)
        temporary_path = ""
        os.chmod(target_path, 0o600)
    except OSError:
        raise HTTPException(500, "无法写入 OSS 配置文件")
    finally:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)
    return existed


def local_ossutil() -> str:
    executable = resolve_ossutil_bin()
    if not executable:
        raise HTTPException(501, "未检测到 ossutil，请在页面点击安装")
    require_oss_config()
    return executable


def local_ossutil_args(executable: str, *args: str) -> List[str]:
    return [executable, "--config-file", OSS_CONFIG_FILE, *args]


def local_rsync() -> str:
    executable = shutil.which("rsync")
    if not executable:
        raise HTTPException(501, "未检测到 rsync")
    return executable


async def remote_ossutil(name: str, port: int) -> str:
    for candidate in OSS_REMOTE_CANDIDATES:
        try:
            command = "test -x %s" % shlex.quote(candidate)
            await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)])
            return candidate
        except HTTPException:
            continue
    raise HTTPException(502, "容器中未找到 ossutil；请确认 NAS 已挂载")


async def check_remote_tools(name: str, port: int, *, tar: bool = False, python3: bool = False, rsync: bool = False) -> str:
    executable = await remote_ossutil(name, port)
    commands = ["command -v setsid >/dev/null"]
    if tar:
        commands.append("command -v tar >/dev/null")
    if python3:
        commands.append("command -v python3 >/dev/null")
    if rsync:
        commands.append("command -v rsync >/dev/null")
    if commands:
        await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(" && ".join(commands))])
    return executable


async def copy_oss_config_to_remote(name: str, port: int, task_id: str) -> str:
    require_oss_config()
    remote_config = "/tmp/itask_gui_oss_%s.conf" % task_id
    with open(OSS_CONFIG_FILE, "rb") as handle:
        contents = handle.read()
    process = await asyncio.create_subprocess_exec(
        *ssh_args(name, port), "sh", "-lc", shlex.quote("umask 077; cat > %s; chmod 600 %s" % (shlex.quote(remote_config), shlex.quote(remote_config))),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env(),
    )
    try:
        _, stderr = await process.communicate(contents)
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise HTTPException(502, stderr.decode(errors="replace").strip() or "无法向容器传送 OSS 临时配置")
    return remote_config


async def run_remote_cleanup(name: str, port: int, command: str, label: str) -> int:
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)], 15)
            return attempt
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                await asyncio.sleep(0.25 * (2 ** (attempt - 1)))
    detail = str(getattr(last_error, "detail", last_error))
    raise RuntimeError("%s清理失败（已重试 3 次）：%s" % (label, detail))


async def await_uninterruptible(coroutine):
    task = asyncio.create_task(coroutine)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def remove_remote_oss_config(name: str, port: int, remote_config: str) -> int:
    command = "rm -f -- %s" % shlex.quote(remote_config)
    return await run_remote_cleanup(name, port, command, "Pod OSS 临时配置")


def append_oss_log(state: Dict[str, Any], result: Dict[str, str]) -> None:
    lines = (result["output"] + "\n" + result["error"]).strip().splitlines()
    if lines:
        state["log"] = (state.get("log", []) + lines)[-40:]


async def run_oss_command(state: Dict[str, Any], args: List[str], timeout: float) -> Dict[str, str]:
    result = await run(args, timeout)
    append_oss_log(state, result)
    return result


def oss_object_size(output: str, user: str, key: str) -> Optional[int]:
    for item in parse_oss_listing(output, user):
        if item.get("key") == key and isinstance(item.get("size"), int) and item["size"] >= 0:
            return item["size"]
    return None


def update_download_progress(
    state: Dict[str, Any], current_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None, complete: bool = False,
) -> None:
    progress = state.setdefault("progress", {
        "bytes": 0, "total": None, "percent": None, "speed": 0, "started_at": None,
    })
    if progress.get("started_at") is None:
        progress["started_at"] = time.time()
    if isinstance(total_bytes, int) and total_bytes >= 0:
        progress["total"] = total_bytes
    if isinstance(current_bytes, int) and current_bytes >= 0:
        progress["bytes"] = current_bytes
    current = int(progress.get("bytes") or 0)
    total = progress.get("total")
    if isinstance(total, int) and total > 0 and current <= total:
        progress["percent"] = min(100, max(0, round(current * 100 / total, 1)))
    else:
        progress["percent"] = None
    elapsed = max(time.time() - float(progress["started_at"]), 0.001)
    progress["speed"] = max(0, int(current / elapsed))
    if complete:
        progress["bytes"] = current
        progress["total"] = current
        progress["percent"] = 100
        progress["speed"] = 0


def regular_file_download_size(path: str) -> int:
    try:
        if not os.path.isfile(path) or os.path.islink(path):
            return 0
        file_stat = os.stat(path, follow_symlinks=False)
        allocated = int(getattr(file_stat, "st_blocks", 0)) * 512
        if 0 < allocated < file_stat.st_size:
            return allocated
        return file_stat.st_size
    except OSError:
        return 0


def local_download_size(path: str) -> int:
    if os.path.isfile(path) and not os.path.islink(path):
        return regular_file_download_size(path)
    temporary_path = path + ".temp"
    if os.path.isfile(temporary_path) and not os.path.islink(temporary_path):
        return regular_file_download_size(temporary_path)
    try:
        if not os.path.isdir(path) or os.path.islink(path):
            return 0
        total = 0
        for current, directories, filenames in os.walk(path, followlinks=False):
            directories[:] = [name for name in directories if not os.path.islink(os.path.join(current, name))]
            for name in filenames:
                total += regular_file_download_size(os.path.join(current, name))
        return total
    except OSError:
        return 0


def remote_download_size_command(path: str) -> str:
    quoted_path = shlex.quote(path)
    quoted_temporary = shlex.quote(path + ".temp")
    return """target={path}
temporary={temporary}
if [ -f "$target" ] && [ ! -L "$target" ]; then
    stat -c '%s %b' -- "$target" | awk '{{ allocated = $2 * 512; print (allocated > 0 && allocated < $1) ? allocated : $1 }}'
elif [ -f "$temporary" ] && [ ! -L "$temporary" ]; then
    stat -c '%s %b' -- "$temporary" | awk '{{ allocated = $2 * 512; print (allocated > 0 && allocated < $1) ? allocated : $1 }}'
elif [ -d "$target" ] && [ ! -L "$target" ]; then
    find "$target" -type f -exec stat -c '%s %b' -- {{}} + | awk '{{ allocated = $2 * 512; total += (allocated > 0 && allocated < $1) ? allocated : $1 }} END {{ print total + 0 }}'
else
    printf 0
fi""".format(path=quoted_path, temporary=quoted_temporary)


async def run_local_download_command(
    state: Dict[str, Any], args: List[str], timeout: float, staged_path: str,
    total_bytes: Optional[int] = None,
) -> Dict[str, str]:
    update_download_progress(state, 0, total_bytes)
    command_task = asyncio.create_task(run_oss_command(state, args, timeout))
    try:
        while not command_task.done():
            await asyncio.wait({command_task}, timeout=0.5)
            current = await asyncio.to_thread(local_download_size, staged_path)
            update_download_progress(state, current)
        result = await command_task
        current = await asyncio.to_thread(local_download_size, staged_path)
        update_download_progress(state, current, complete=True)
        return result
    except BaseException:
        if not command_task.done():
            command_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await command_task
        raise


async def run_remote_download_command(
    state: Dict[str, Any], name: str, port: int, remote_work: str,
    command: str, timeout: float, staged_path: str, total_bytes: Optional[int] = None,
) -> Dict[str, str]:
    update_download_progress(state, 0, total_bytes)
    command_task = asyncio.create_task(
        run_tracked_remote_command(state, name, port, remote_work, command, timeout)
    )
    stat_command = remote_download_size_command(staged_path)
    try:
        while not command_task.done():
            await asyncio.wait({command_task}, timeout=0.75)
            if command_task.done():
                break
            try:
                result = await run(
                    ssh_args(name, port) + ["sh", "-lc", shlex.quote(stat_command)], 10,
                )
                update_download_progress(state, int(result["output"].strip() or "0"))
            except (HTTPException, ValueError):
                pass
        result = await command_task
        try:
            stat_result = await run(
                ssh_args(name, port) + ["sh", "-lc", shlex.quote(stat_command)], 10,
            )
            current = int(stat_result["output"].strip() or "0")
        except (HTTPException, ValueError):
            current = int(state.get("progress", {}).get("bytes") or 0)
        update_download_progress(state, current, complete=True)
        return result
    except BaseException:
        if not command_task.done():
            command_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await command_task
        raise


def rsync_args(rsync: str, source: str, destination: str, excludes: List[str], conflict_policy: str = "overwrite") -> List[str]:
    args = [rsync, "-a"]
    for pattern in excludes:
        args.extend(["--exclude", pattern])
    if conflict_policy == "skip":
        args.append("--ignore-existing")
    return args + [source.rstrip("/") + "/", destination.rstrip("/") + "/"]


def _excluded(relative_path: str, excludes: List[str]) -> bool:
    normalized = relative_path.replace(os.sep, "/").strip("/")
    return any(fnmatch(normalized, pattern) or fnmatch(posixpath.basename(normalized), pattern) for pattern in excludes)


def remote_source_contains(source: str, sensitive_path: str) -> bool:
    try:
        return posixpath.commonpath([source, sensitive_path]) == source
    except ValueError:
        return False


async def run_local_io(function, *args):
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # TemporaryDirectory must stay alive until the worker has stopped using it.
        with contextlib.suppress(Exception):
            await worker
        raise


def detect_local_source(path: str, requested_kind: str) -> tuple[str, str]:
    resolved = safe_local_path(path)
    try:
        source_stat = os.stat(resolved)
    except FileNotFoundError:
        raise HTTPException(400, "上传来源不存在：%s" % resolved)
    except OSError as exc:
        raise HTTPException(400, "无法检查上传来源：%s" % exc)
    actual = "file" if stat.S_ISREG(source_stat.st_mode) else "directory" if stat.S_ISDIR(source_stat.st_mode) else "other"
    if actual == "other":
        raise HTTPException(400, "上传来源必须是普通文件或目录")
    if requested_kind != "auto" and requested_kind != actual:
        raise HTTPException(400, "上传来源类型与 source_kind 不一致")
    return resolved, actual


def validate_local_tree(root: str, reject_unsupported: bool = True) -> Dict[str, int]:
    files = 0
    total_bytes = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            path = os.path.join(current, name)
            item_stat = os.lstat(path)
            if stat.S_ISLNK(item_stat.st_mode) or not (stat.S_ISDIR(item_stat.st_mode) or stat.S_ISREG(item_stat.st_mode)):
                if reject_unsupported:
                    raise HTTPException(400, "目录包含符号链接或特殊文件")
                continue
            if stat.S_ISREG(item_stat.st_mode):
                files += 1
                total_bytes += item_stat.st_size
    return {"files": files, "bytes": total_bytes}


def pack_local_directory(source: str, archive: str, excludes: List[str]) -> Dict[str, int]:
    files = 0
    total_bytes = 0
    with tarfile.open(archive, "w") as output:
        for current, directories, filenames in os.walk(source, topdown=True, followlinks=False):
            relative_root = os.path.relpath(current, source)
            kept_directories = []
            for name in directories:
                relative = name if relative_root == "." else os.path.join(relative_root, name)
                item_stat = os.lstat(os.path.join(current, name))
                if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(item_stat.st_mode):
                    raise HTTPException(400, "上传目录包含符号链接或特殊文件")
                if not _excluded(relative, excludes):
                    kept_directories.append(name)
                    output.add(os.path.join(current, name), arcname=relative.replace(os.sep, "/"), recursive=False)
            directories[:] = kept_directories
            for name in filenames:
                relative = name if relative_root == "." else os.path.join(relative_root, name)
                path = os.path.join(current, name)
                item_stat = os.lstat(path)
                if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
                    raise HTTPException(400, "上传目录包含符号链接或特殊文件")
                if _excluded(relative, excludes):
                    continue
                output.add(path, arcname=relative.replace(os.sep, "/"), recursive=False)
                files += 1
                total_bytes += item_stat.st_size
    return {"files": files, "bytes": total_bytes}


def validate_archive_member(member: tarfile.TarInfo, seen: set[str]) -> str:
    name = member.name
    parts = name.replace("\\", "/").split("/")
    normalized = posixpath.normpath(name)
    if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part == ".." for part in parts) or normalized in {"", ".."} or normalized.startswith("../"):
        raise HTTPException(400, "归档包含不安全路径")
    if normalized == ".":
        if not member.isdir():
            raise HTTPException(400, "归档包含不安全路径")
        return normalized
    if normalized in seen:
        raise HTTPException(400, "归档包含重复路径")
    seen.add(normalized)
    if not (member.isdir() or member.isreg()):
        raise HTTPException(400, "归档包含符号链接、硬链接或设备文件")
    return normalized


def safe_extract_local_tar(archive: str, destination: str) -> Dict[str, int]:
    declared_bytes = 0
    member_count = 0
    seen: set[str] = set()
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            validate_archive_member(member, seen)
            member_count += 1
            if member_count > OSS_ARCHIVE_MAX_MEMBERS:
                raise HTTPException(400, "归档成员数量超过安全上限（%d）" % OSS_ARCHIVE_MAX_MEMBERS)
            if member.isreg():
                declared_bytes += member.size
                if declared_bytes > OSS_ARCHIVE_MAX_BYTES:
                    raise HTTPException(400, "归档声明展开大小超过安全上限（%d 字节）" % OSS_ARCHIVE_MAX_BYTES)

    files = 0
    actual_bytes = 0
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            normalized = posixpath.normpath(member.name)
            if normalized == ".":
                continue
            target = os.path.join(destination, *normalized.split("/"))
            if member.isdir():
                os.makedirs(target, mode=member.mode & 0o777, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise HTTPException(400, "无法读取归档成员")
            with open(target, "xb") as output:
                while True:
                    block = extracted.read(1024 * 1024)
                    if not block:
                        break
                    actual_bytes += len(block)
                    if actual_bytes > OSS_ARCHIVE_MAX_BYTES:
                        raise HTTPException(400, "归档实际展开大小超过安全上限（%d 字节）" % OSS_ARCHIVE_MAX_BYTES)
                    output.write(block)
            os.chmod(target, member.mode & 0o777)
            files += 1
    validate_local_tree(destination)
    return {"files": files, "bytes": actual_bytes}


def merge_local_file(staged: str, destination_path: str, conflict_policy: str) -> bool:
    original_destination = os.path.abspath(os.path.expanduser(destination_path))
    if os.path.lexists(original_destination) and stat.S_ISLNK(os.lstat(original_destination).st_mode):
        raise HTTPException(409, "目标路径已存在且不是普通文件，无法覆盖")
    destination = safe_local_path(destination_path)
    parent = safe_local_directory(os.path.dirname(destination) or OSS_SYNC_ROOT)
    os.makedirs(parent, exist_ok=True)
    if not stat.S_ISREG(os.lstat(staged).st_mode):
        raise HTTPException(400, "下载对象不是普通文件")
    if os.path.lexists(destination):
        destination_stat = os.lstat(destination)
        if not stat.S_ISREG(destination_stat.st_mode):
            raise HTTPException(409, "目标路径已存在且不是普通文件，无法覆盖")
        if conflict_policy == "skip":
            return False
    descriptor, temporary_path = tempfile.mkstemp(prefix=".itask-oss-", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as output, open(staged, "rb") as source:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
        temporary_path = ""
    finally:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)
    return True


def validate_local_merge_target(staged: str, destination: str) -> str:
    target = safe_local_directory(destination)
    if os.path.lexists(target):
        target_stat = os.lstat(target)
        if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(target_stat.st_mode):
            raise HTTPException(409, "目标路径已存在且不是普通目录，无法合并")
    else:
        os.makedirs(target)
    for current, directories, filenames in os.walk(staged, followlinks=False):
        relative_root = os.path.relpath(current, staged)
        for name in directories + filenames:
            relative = name if relative_root == "." else os.path.join(relative_root, name)
            incoming = os.lstat(os.path.join(current, name))
            existing = os.path.join(target, relative)
            if not os.path.lexists(existing):
                continue
            existing_stat = os.lstat(existing)
            if stat.S_ISLNK(existing_stat.st_mode) or not (stat.S_ISDIR(existing_stat.st_mode) or stat.S_ISREG(existing_stat.st_mode)):
                raise HTTPException(409, "目标目录包含符号链接或特殊文件，无法安全合并")
            if stat.S_ISDIR(incoming.st_mode) != stat.S_ISDIR(existing_stat.st_mode):
                raise HTTPException(409, "目标目录存在同名类型冲突")
    return target


REMOTE_SAFE_TAR_SCRIPT = r'''import os, stat, sys, tarfile
archive, destination, max_members, max_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(destination, exist_ok=True)
def checked(member, seen):
    name = member.name; normalized = os.path.normpath(name.replace("\\", "/")); parts = name.replace("\\", "/").split("/")
    if not name or "\x00" in name or "\\" in name or name.startswith("/") or ".." in parts or normalized in ("", "..") or normalized.startswith("../"):
        raise SystemExit("归档包含不安全路径")
    if normalized == ".":
        if not member.isdir(): raise SystemExit("归档包含不安全路径")
        return normalized
    if normalized in seen: raise SystemExit("归档包含重复路径")
    seen.add(normalized)
    if not (member.isdir() or member.isreg()): raise SystemExit("归档包含符号链接、硬链接或设备文件")
    return normalized
seen = set(); count = 0; declared = 0
with tarfile.open(archive, "r:*") as source:
    for member in source:
        checked(member, seen); count += 1
        if count > max_members: raise SystemExit("归档成员数量超过安全上限（%d）" % max_members)
        if member.isreg():
            declared += member.size
            if declared > max_bytes: raise SystemExit("归档声明展开大小超过安全上限（%d 字节）" % max_bytes)
actual = 0
with tarfile.open(archive, "r:*") as source:
    for member in source:
        normalized = os.path.normpath(member.name.replace("\\", "/"))
        if normalized == ".": continue
        target = os.path.join(destination, *normalized.split("/"))
        if member.isdir():
            os.makedirs(target, mode=member.mode & 0o777, exist_ok=True); continue
        os.makedirs(os.path.dirname(target), exist_ok=True); source_file = source.extractfile(member)
        if source_file is None: raise SystemExit("无法读取归档成员")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, member.mode & 0o777)
        with os.fdopen(descriptor, "wb") as output:
            while True:
                block = source_file.read(1024 * 1024)
                if not block: break
                actual += len(block)
                if actual > max_bytes: raise SystemExit("归档实际展开大小超过安全上限（%d 字节）" % max_bytes)
                output.write(block)
for current, directories, files in os.walk(destination, followlinks=False):
    for name in directories + files:
        mode = os.lstat(os.path.join(current, name)).st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)): raise SystemExit("解包结果包含符号链接或特殊文件")
'''


REMOTE_RESOLVE_UPLOAD_SOURCE_SCRIPT = r'''import os, sys
path = os.path.realpath(os.path.normpath(sys.argv[1]))
if not os.path.exists(path): raise SystemExit("上传来源不存在")
print(path)
'''


REMOTE_VALIDATE_UPLOAD_SCRIPT = r'''import os, stat, sys
path, requested, strict = sys.argv[1], sys.argv[2], sys.argv[3] == "strict"
try: source = os.lstat(path)
except FileNotFoundError: raise SystemExit("上传来源不存在")
except OSError as exc: raise SystemExit("无法检查上传来源：%s" % exc)
actual = "file" if stat.S_ISREG(source.st_mode) else "directory" if stat.S_ISDIR(source.st_mode) else "other"
if actual == "other": raise SystemExit("上传来源必须是普通文件或目录")
if requested != "auto" and requested != actual: raise SystemExit("上传来源类型与 source_kind 不一致")
files = 1 if actual == "file" else 0; size = source.st_size if actual == "file" else 0
if actual == "directory":
    def walk_error(exc): raise exc
    try:
        for current, directories, filenames in os.walk(path, followlinks=False, onerror=walk_error):
            for name in directories + filenames:
                item = os.lstat(os.path.join(current, name)); item_mode = item.st_mode
                if stat.S_ISLNK(item_mode) or not (stat.S_ISDIR(item_mode) or stat.S_ISREG(item_mode)):
                    if strict: raise SystemExit("Pod 上传目录包含符号链接、FIFO、socket、设备或其他特殊文件")
                    continue
                if stat.S_ISREG(item_mode): files += 1; size += item.st_size
    except OSError as exc: raise SystemExit("无法完整检查 Pod 上传目录：%s" % exc)
print("ITASK_STATS\t%s\t%d\t%d" % (actual, files, size))
'''


REMOTE_VERIFY_TREE_SCRIPT = r'''import os, stat, sys
for current, directories, files in os.walk(sys.argv[1], followlinks=False):
    for name in directories + files:
        mode = os.lstat(os.path.join(current, name)).st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)): raise SystemExit("下载内容包含符号链接或特殊文件")
'''


REMOTE_VALIDATE_MERGE_SCRIPT = r'''import os, stat, sys
source, target = sys.argv[1], sys.argv[2]
if os.path.lexists(target):
    mode = os.lstat(target).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode): raise SystemExit("目标路径已存在且不是普通目录，无法合并")
else: os.makedirs(target)
for current, directories, files in os.walk(source, followlinks=False):
    relative_root = os.path.relpath(current, source)
    for name in directories + files:
        relative = name if relative_root == "." else os.path.join(relative_root, name)
        incoming = os.lstat(os.path.join(current, name)).st_mode; existing = os.path.join(target, relative)
        if not os.path.lexists(existing): continue
        mode = os.lstat(existing).st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)): raise SystemExit("目标目录包含符号链接或特殊文件，无法安全合并")
        if stat.S_ISDIR(incoming) != stat.S_ISDIR(mode): raise SystemExit("目标目录存在同名类型冲突")
'''


def validate_transfer_request(request: OssUnifiedTransferRequest) -> None:
    if request.direction == "upload":
        if request.oss_key:
            raise HTTPException(400, "上传目标由来源名称自动生成，请勿指定 oss_key")
        if request.source_kind not in {"auto", "file", "directory"}:
            raise HTTPException(400, "上传 source_kind 必须是 auto、file 或 directory")
        if request.extract_download:
            raise HTTPException(400, "上传不能设置 extract_download")
    else:
        if not request.oss_key:
            raise HTTPException(400, "下载必须指定 oss_key")
        if request.source_kind not in {"object", "prefix"}:
            raise HTTPException(400, "下载 source_kind 必须明确为 object 或 prefix")
        if request.package_upload:
            raise HTTPException(400, "下载不能设置 package_upload")
        if request.source_kind == "prefix" and request.extract_download:
            raise HTTPException(400, "前缀下载不能设置 extract_download")


def remote_exclude_args(excludes: List[str]) -> str:
    return " ".join("--exclude=%s" % shlex.quote(pattern) for pattern in excludes)


def remote_rsync_flags(conflict_policy: str, excludes: List[str]) -> str:
    flags = ["-a"]
    if conflict_policy == "skip":
        flags.append("--ignore-existing")
    for pattern in excludes:
        flags.extend(["--exclude", shlex.quote(pattern)])
    return " ".join(flags)


async def run_oss_local_transfer(task_id: str, request: OssUnifiedTransferRequest) -> None:
    state = oss_states[task_id]
    try:
        state.update(status="running", message="正在检查本机工具", phase="detecting")
        validate_transfer_request(request)
        ossutil = local_ossutil()
        target_uri = oss_uri(request.user, request.oss_key) if request.direction == "download" else ""
        if request.direction == "download":
            state["oss_key"] = request.oss_key
        with tempfile.TemporaryDirectory(prefix="itask-gui-oss-") as temporary:
            checkpoint_dir = os.path.join(temporary, "checkpoint")
            output_dir = os.path.join(temporary, "output")
            os.mkdir(checkpoint_dir)
            os.mkdir(output_dir)
            transfer_options = ["--checkpoint-dir", checkpoint_dir, "--output-dir", output_dir]
            if request.direction == "upload":
                source, source_kind = await run_local_io(detect_local_source, request.filesystem_path, request.source_kind)
                upload_key = automatic_upload_key(request.filesystem_path, source_kind, request.package_upload)
                target_uri = oss_uri(request.user, upload_key)
                state.update(source_kind=source_kind, oss_key=upload_key)
                if source_kind == "file":
                    source_stat = await run_local_io(os.lstat, source)
                    state["statistics"].update(files=1, bytes=source_stat.st_size)
                    state.update(message="正在上传普通文件", phase="uploading")
                    await run_oss_command(state, local_ossutil_args(ossutil, "cp", source, target_uri, "-f", *transfer_options), 3600)
                elif request.package_upload:
                    archive = os.path.join(temporary, "archive.tar")
                    state.update(message="正在打包本机目录", phase="packing")
                    state["statistics"].update(await run_local_io(pack_local_directory, source, archive, request.excludes))
                    state.update(message="正在上传目录归档", phase="uploading")
                    await run_oss_command(state, local_ossutil_args(ossutil, "cp", archive, target_uri, "-f", *transfer_options), 3600)
                else:
                    state["statistics"].update(await run_local_io(validate_local_tree, source, False))
                    state.update(message="正在直接同步本机目录", phase="uploading")
                    args = local_ossutil_args(
                        ossutil, "sync", source.rstrip("/") + "/", target_uri.rstrip("/") + "/",
                        "--disable-all-symlink", "--disable-ignore-error", "-f", *transfer_options,
                    )
                    for pattern in request.excludes:
                        args.extend(["--exclude", pattern])
                    await run_oss_command(state, args, 3600)
            elif request.source_kind == "object":
                staged_object = os.path.join(temporary, "object")
                listing = await run(local_ossutil_args(ossutil, "ls", target_uri), 60)
                total_bytes = oss_object_size(listing["output"], request.user, request.oss_key)
                state.update(message="正在下载 OSS 对象", phase="downloading")
                await run_local_download_command(
                    state,
                    local_ossutil_args(ossutil, "cp", target_uri, staged_object, "-f", *transfer_options),
                    3600,
                    staged_object,
                    total_bytes,
                )
                if not os.path.isfile(staged_object) or os.path.islink(staged_object):
                    raise HTTPException(400, "下载对象不是普通文件")
                state.update(message="正在校验下载对象", phase="verifying")
                if request.extract_download:
                    extracted = os.path.join(temporary, "extracted")
                    os.mkdir(extracted)
                    state.update(message="正在安全解包", phase="extracting")
                    state["statistics"].update(await run_local_io(safe_extract_local_tar, staged_object, extracted))
                    target = await run_local_io(validate_local_merge_target, extracted, request.filesystem_path)
                    state.update(message="正在合并到本机目录", phase="merging")
                    await run_oss_command(state, rsync_args(local_rsync(), extracted, target, request.excludes, request.conflict_policy), 3600)
                else:
                    state["statistics"].update(files=1, bytes=os.path.getsize(staged_object))
                    state.update(message="正在写入目标文件", phase="merging")
                    if not await run_local_io(merge_local_file, staged_object, request.filesystem_path, request.conflict_policy):
                        state["statistics"]["skipped"] = 1
            else:
                staged_tree = os.path.join(temporary, "prefix")
                os.mkdir(staged_tree)
                state.update(message="正在递归下载 OSS 前缀", phase="downloading")
                args = local_ossutil_args(
                    ossutil, "sync", target_uri.rstrip("/") + "/", staged_tree.rstrip("/") + "/",
                    "--disable-ignore-error", "-f", *transfer_options,
                )
                for pattern in request.excludes:
                    args.extend(["--exclude", pattern])
                await run_local_download_command(state, args, 3600, staged_tree)
                state.update(message="正在校验下载目录", phase="verifying")
                state["statistics"].update(await run_local_io(validate_local_tree, staged_tree))
                target = await run_local_io(validate_local_merge_target, staged_tree, request.filesystem_path)
                state.update(message="正在合并到本机目录", phase="merging")
                await run_oss_command(state, rsync_args(local_rsync(), staged_tree, target, request.excludes, request.conflict_policy), 3600)
        state.update(status="done", message="传输完成", phase="done", finished_at=time.time())
    except asyncio.CancelledError:
        state.update(status="cancelled", message="传输已取消", phase="cancelled", finished_at=time.time())
        raise
    except Exception as exc:
        state.update(status="error", message=str(getattr(exc, "detail", exc)), phase="error", finished_at=time.time())
    finally:
        oss_tasks.pop(task_id, None)



def update_remote_statistics(state: Dict[str, Any], result: Dict[str, str]) -> None:
    for line in result["output"].splitlines():
        if not line.startswith("ITASK_STATS\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        try:
            state["source_kind"] = parts[1]
            state["statistics"].update(files=int(parts[2]), bytes=int(parts[3]))
        except ValueError:
            continue


async def remove_remote_work(name: str, port: int, remote_work: str) -> int:
    command = "rm -rf -- %s" % shlex.quote(remote_work)
    return await run_remote_cleanup(name, port, command, "Pod OSS 临时工作目录")


async def run_tracked_remote_command(
    state: Dict[str, Any], name: str, port: int, remote_work: str, command: str, timeout: float,
) -> Dict[str, str]:
    pid_file = "%s/active.pgid" % remote_work
    wrapper = """set -u
pid_file={pid_file}
rm -f -- "$pid_file"
setsid bash -lc {command} &
pgid=$!
printf '%s\n' "$pgid" > "$pid_file"
terminate_group() {{ kill -TERM -- "-$pgid" 2>/dev/null || true; }}
trap terminate_group HUP INT TERM
status=0
wait "$pgid" || status=$?
rm -f -- "$pid_file"
trap - HUP INT TERM
exit "$status"
""".format(pid_file=shlex.quote(pid_file), command=shlex.quote(command))
    try:
        return await run_oss_command(
            state,
            ssh_args(name, port) + ["bash", "-lc", shlex.quote(wrapper)],
            timeout,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        try:
            await await_uninterruptible(
                terminate_tracked_remote_command(name, port, remote_work, state["task_id"])
            )
        except Exception as termination_exc:
            raise HTTPException(
                502,
                "%s；同时无法确认 Pod 远端进程已终止：%s"
                % (str(getattr(exc, "detail", exc)), str(getattr(termination_exc, "detail", termination_exc))),
            )
        raise


async def terminate_tracked_remote_command(name: str, port: int, remote_work: str, task_id: str) -> None:
    pid_file = "%s/active.pgid" % remote_work
    marker = task_id
    command = """set -eu
pid_file={pid_file}
count=0
while [ ! -f "$pid_file" ] && [ "$count" -lt 5 ]; do sleep 0.2; count=$((count + 1)); done
[ -f "$pid_file" ] || exit 0
IFS= read -r pgid < "$pid_file"
case "$pgid" in ''|*[!0-9]*) echo '无效的远端进程组记录' >&2; exit 74;; esac
args=$(ps -o args= -p "$pgid" 2>/dev/null || true)
case "$args" in *{marker}*) :;; *) echo '远端进程身份校验失败' >&2; exit 75;; esac
sid=$(ps -o sid= -p "$pgid" 2>/dev/null | tr -d ' ')
[ "$sid" = "$pgid" ] || {{ echo '远端进程组校验失败' >&2; exit 76; }}
kill -TERM -- "-$pgid" 2>/dev/null || true
count=0
while kill -0 "$pgid" 2>/dev/null && [ "$count" -lt 20 ]; do sleep 0.25; count=$((count + 1)); done
if kill -0 "$pgid" 2>/dev/null; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi
rm -f -- "$pid_file"
""".format(pid_file=shlex.quote(pid_file), marker=shlex.quote(marker))
    await run_remote_cleanup(name, port, command, "Pod OSS 远端进程")


async def run_oss_remote_transfer(task_id: str, name: str, request: OssUnifiedTransferRequest) -> None:
    state = oss_states[task_id]
    port: Optional[int] = None
    remote_config = "/tmp/itask_gui_oss_%s.conf" % task_id
    remote_work = "/tmp/itask_gui_oss_work_%s" % task_id
    transfer_status = "error"
    transfer_message = "传输未完成"
    was_cancelled = False
    termination_error: Optional[str] = None
    try:
        state.update(status="running", message="正在检查 Pod 环境", phase="detecting")
        validate_transfer_request(request)
        filesystem_path = safe_remote_directory(request.filesystem_path)
        require_oss_config()
        port = require_tunnel(name)
        if request.direction == "upload" and any(
            remote_source_contains(filesystem_path, temporary)
            for temporary in (remote_config, remote_work)
        ):
            raise HTTPException(400, "Pod 上传来源不能包含当前任务的临时目录，请选择更具体的路径")
        need_tar = request.direction == "upload" and request.package_upload
        need_python = request.direction == "upload" or (request.direction == "download" and (request.source_kind == "prefix" or request.extract_download))
        need_merge_tools = request.direction == "download" and (request.source_kind == "prefix" or request.extract_download)
        remote_tool = await check_remote_tools(name, port, tar=need_tar, python3=need_python, rsync=need_merge_tools)
        if request.direction == "upload":
            resolve_command = "python3 -c %s %s" % (
                shlex.quote(REMOTE_RESOLVE_UPLOAD_SOURCE_SCRIPT),
                shlex.quote(filesystem_path),
            )
            resolved_result = await run(
                ssh_args(name, port) + ["sh", "-lc", shlex.quote(resolve_command)], 60,
            )
            filesystem_path = safe_remote_directory(resolved_result["output"])
            if any(
                remote_source_contains(filesystem_path, temporary)
                for temporary in (remote_config, remote_work)
            ):
                raise HTTPException(400, "Pod 上传来源不能包含当前任务的临时目录，请选择更具体的路径")
        target_uri = oss_uri(request.user, request.oss_key) if request.direction == "download" else ""
        if request.direction == "download":
            state["oss_key"] = request.oss_key
        remote_config = await copy_oss_config_to_remote(name, port, task_id)
        await run(
            ssh_args(name, port) + ["sh", "-lc", shlex.quote(
                "umask 077; rm -rf -- %s; mkdir -- %s; mkdir -- %s/checkpoint %s/output"
                % (shlex.quote(remote_work), shlex.quote(remote_work), shlex.quote(remote_work), shlex.quote(remote_work))
            )],
            30,
        )
        quoted_tool = shlex.quote(remote_tool)
        quoted_config = shlex.quote(remote_config)
        quoted_work = shlex.quote(remote_work)
        quoted_path = shlex.quote(filesystem_path)
        quoted_uri = shlex.quote(target_uri) if target_uri else ""
        quoted_checkpoint = shlex.quote("%s/checkpoint" % remote_work)
        quoted_output = shlex.quote("%s/output" % remote_work)
        transfer_options = "--checkpoint-dir %s --output-dir %s" % (quoted_checkpoint, quoted_output)
        if request.direction == "upload":
            state.update(message="正在校验 Pod 上传来源", phase="verifying")
            validation = "python3 -c %s %s %s %s" % (
                shlex.quote(REMOTE_VALIDATE_UPLOAD_SCRIPT), quoted_path,
                shlex.quote(request.source_kind),
                "strict" if request.package_upload else "lenient",
            )
            result = await run_tracked_remote_command(state, name, port, remote_work, validation, 300)
            update_remote_statistics(state, result)
            actual = state["source_kind"]
            upload_key = automatic_upload_key(filesystem_path, actual, request.package_upload)
            target_uri = oss_uri(request.user, upload_key)
            quoted_uri = shlex.quote(target_uri)
            state["oss_key"] = upload_key
            if request.package_upload:
                if actual != "directory":
                    raise HTTPException(400, "只有目录可以打包上传")
                state.update(message="正在打包 Pod 目录", phase="packing")
                command = "set -eu; tar -C %s %s -cf %s/archive.tar ." % (
                    quoted_path, remote_exclude_args(request.excludes), quoted_work,
                )
                await run_tracked_remote_command(state, name, port, remote_work, command, 1800)
                state.update(message="正在上传 Pod 目录归档", phase="uploading")
                command = "%s --config-file %s cp %s/archive.tar %s -f %s" % (
                    quoted_tool, quoted_config, quoted_work, quoted_uri, transfer_options,
                )
                await run_tracked_remote_command(state, name, port, remote_work, command, 3600)
            elif actual == "file":
                state.update(message="正在上传 Pod 普通文件", phase="uploading")
                command = "%s --config-file %s cp %s %s -f %s" % (
                    quoted_tool, quoted_config, quoted_path, quoted_uri, transfer_options,
                )
                await run_tracked_remote_command(state, name, port, remote_work, command, 3600)
            else:
                state.update(message="正在直接同步 Pod 目录", phase="uploading")
                sync_args = " ".join("--exclude %s" % shlex.quote(pattern) for pattern in request.excludes)
                command = (
                    "%s --config-file %s sync %s/ %s/ "
                    "--disable-all-symlink --disable-ignore-error -f %s %s"
                ) % (
                    quoted_tool, quoted_config, quoted_path,
                    shlex.quote(target_uri.rstrip("/")), transfer_options, sync_args,
                )
                await run_tracked_remote_command(state, name, port, remote_work, command, 3600)
        elif request.source_kind == "object" and not request.extract_download:
            listing_command = "%s --config-file %s ls %s" % (
                quoted_tool, quoted_config, quoted_uri,
            )
            listing = await run(
                ssh_args(name, port) + ["sh", "-lc", shlex.quote(listing_command)], 60,
            )
            total_bytes = oss_object_size(listing["output"], request.user, request.oss_key)
            state.update(message="正在下载 OSS 对象到 Pod 临时目录", phase="downloading")
            command = "%s --config-file %s cp %s %s/object -f %s" % (
                quoted_tool, quoted_config, quoted_uri, quoted_work, transfer_options,
            )
            await run_remote_download_command(
                state, name, port, remote_work, command, 3600,
                "%s/object" % remote_work, total_bytes,
            )
            state.update(message="正在校验下载对象", phase="verifying")
            command = "set -eu; [ -f %s/object ] && [ ! -L %s/object ]; bytes=$(stat -c %%s -- %s/object); printf 'ITASK_STATS\\tobject\\t1\\t%%s\\n' \"$bytes\"" % (quoted_work, quoted_work, quoted_work)
            result = await run_tracked_remote_command(state, name, port, remote_work, command, 60)
            update_remote_statistics(state, result)
            state.update(message="正在写入 Pod 目标文件", phase="merging")
            parent = posixpath.dirname(filesystem_path) or "/"
            skip_action = "printf ITASK_SKIPPED; exit 0" if request.conflict_policy == "skip" else ":"
            command = (
                "set -eu\ndestination={destination}\ntemporary=''\n"
                "cleanup() {{ [ -z \"$temporary\" ] || rm -f -- \"$temporary\"; }}\ntrap cleanup EXIT HUP INT TERM\n"
                "mkdir -p -- {parent}\n"
                "if [ -L \"$destination\" ] || {{ [ -e \"$destination\" ] && [ ! -f \"$destination\" ]; }}; then echo '目标路径已存在且不是普通文件，无法覆盖' >&2; exit 73; fi\n"
                "if [ -e \"$destination\" ]; then {skip}; fi\n"
                "temporary=$(mktemp {parent}/.itask-oss.XXXXXX)\ncp -- {work}/object \"$temporary\"\nmv -f -- \"$temporary\" \"$destination\"\n"
                "temporary=''\ntrap - EXIT HUP INT TERM"
            ).format(destination=quoted_path, parent=shlex.quote(parent), skip=skip_action, work=quoted_work)
            result = await run_tracked_remote_command(state, name, port, remote_work, command, 300)
            state["statistics"]["skipped"] = 1 if "ITASK_SKIPPED" in result["output"] else 0
        else:
            staged = "%s/staged" % remote_work
            quoted_staged = shlex.quote(staged)
            if request.source_kind == "object":
                listing_command = "%s --config-file %s ls %s" % (
                    quoted_tool, quoted_config, quoted_uri,
                )
                listing = await run(
                    ssh_args(name, port) + ["sh", "-lc", shlex.quote(listing_command)], 60,
                )
                total_bytes = oss_object_size(listing["output"], request.user, request.oss_key)
                state.update(message="正在下载 OSS 归档到 Pod 临时目录", phase="downloading")
                command = "%s --config-file %s cp %s %s/archive.tar -f %s" % (
                    quoted_tool, quoted_config, quoted_uri, quoted_work, transfer_options,
                )
                await run_remote_download_command(
                    state, name, port, remote_work, command, 3600,
                    "%s/archive.tar" % remote_work, total_bytes,
                )
                state.update(message="正在安全解包 Pod 归档", phase="extracting")
                command = "python3 -c %s %s/archive.tar %s %d %d" % (
                    shlex.quote(REMOTE_SAFE_TAR_SCRIPT), quoted_work, quoted_staged,
                    OSS_ARCHIVE_MAX_MEMBERS, OSS_ARCHIVE_MAX_BYTES,
                )
                await run_tracked_remote_command(state, name, port, remote_work, command, 1800)
            else:
                state.update(message="正在递归下载 OSS 前缀到 Pod 临时目录", phase="downloading")
                sync_args = " ".join("--exclude %s" % shlex.quote(pattern) for pattern in request.excludes)
                command = (
                    "mkdir -- %s && %s --config-file %s sync %s/ %s/ "
                    "--disable-ignore-error -f %s %s"
                ) % (
                    quoted_staged, quoted_tool, quoted_config,
                    shlex.quote(target_uri.rstrip("/")), quoted_staged, transfer_options, sync_args,
                )
                await run_remote_download_command(
                    state, name, port, remote_work, command, 3600, staged,
                )
            state.update(message="正在校验 Pod 下载目录", phase="verifying")
            command = (
                "set -eu\npython3 -c {script} {staged}\n"
                "files=$(find {staged} -type f -printf . | wc -c | tr -d ' '); bytes=$(find {staged} -type f -printf '%s\\n' | awk '{{total += $1}} END {{print total + 0}}'); "
                "printf 'ITASK_STATS\\t{kind}\\t%s\\t%s\\n' \"$files\" \"$bytes\""
            ).format(script=shlex.quote(REMOTE_VERIFY_TREE_SCRIPT), staged=quoted_staged, kind=request.source_kind)
            result = await run_tracked_remote_command(state, name, port, remote_work, command, 300)
            update_remote_statistics(state, result)
            state.update(message="正在合并到 Pod 目录", phase="merging")
            command = "python3 -c %s %s %s && rsync %s %s/ %s/" % (
                shlex.quote(REMOTE_VALIDATE_MERGE_SCRIPT), quoted_staged, quoted_path,
                remote_rsync_flags(request.conflict_policy, request.excludes), quoted_staged, quoted_path,
            )
            await run_tracked_remote_command(state, name, port, remote_work, command, 3600)
        transfer_status = "done"
        transfer_message = "传输完成"
    except asyncio.CancelledError:
        was_cancelled = True
        transfer_status = "cancelled"
        transfer_message = "传输已取消"
        state.update(status="running", message="正在终止 Pod 传输进程", phase="cancelling")
        if port is not None:
            try:
                await await_uninterruptible(terminate_tracked_remote_command(name, port, remote_work, task_id))
            except Exception as exc:
                termination_error = str(getattr(exc, "detail", exc))
    except Exception as exc:
        transfer_status = "error"
        transfer_message = str(getattr(exc, "detail", exc))
    finally:
        cleanup_errors = []
        cleanup_attempts: Dict[str, int] = {}
        if port is not None:
            state.update(status="running", message="正在清理 Pod 临时文件", phase="cleaning")
            for label, cleanup in (
                ("work", remove_remote_work(name, port, remote_work)),
                ("config", remove_remote_oss_config(name, port, remote_config)),
            ):
                try:
                    cleanup_attempts[label] = await await_uninterruptible(cleanup)
                except Exception as exc:
                    cleanup_errors.append(str(getattr(exc, "detail", exc)))
        if termination_error:
            cleanup_errors.insert(0, "远端进程终止未确认：%s" % termination_error)
        state["cleanup"] = {
            "status": "error" if cleanup_errors else "done",
            "attempts": cleanup_attempts,
            "errors": cleanup_errors,
        }
        if cleanup_errors:
            state.update(
                status="error",
                phase="cleanup_error",
                message="%s，但 Pod 清理未完成，可能存在进程、临时文件或凭据残留：%s"
                % (transfer_message, "；".join(cleanup_errors)),
                finished_at=time.time(),
            )
            state["log"] = (state.get("log", []) + ["CLEANUP ERROR: %s" % error for error in cleanup_errors])[-40:]
        else:
            state.update(
                status=transfer_status,
                phase=transfer_status,
                message=transfer_message,
                finished_at=time.time(),
            )
        oss_tasks.pop(task_id, None)
    if was_cancelled:
        raise asyncio.CancelledError




@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIST_DIR, "index.html"))


def dependency_item(dependency_id: str, label: str, feature: str, ready: bool, detail: str) -> Dict[str, Any]:
    return {"id": dependency_id, "label": label, "feature": feature, "ready": ready, "detail": detail}


def ossutil_version_line(result: Dict[str, str]) -> str:
    for line in (result.get("output", "") + "\n" + result.get("error", "")).splitlines():
        if line.strip():
            return line.strip()
    return ""


async def ossutil_readiness_item() -> Dict[str, Any]:
    platform_info = ossutil_platform_info()
    target = display_home_path(OSSUTIL_MANAGED_BIN)
    executable = resolve_ossutil_bin()
    base = {
        "id": "ossutil",
        "label": "ossutil",
        "feature": "本机 OSS",
        "installed": bool(executable),
        "usable": False,
        "state": "missing",
        "code": "not_installed",
        "installable": platform_info["supported"],
        "path": executable,
        "install_target": target,
        "platform": platform_info["platform"],
        "platform_label": platform_info["label"],
        "version": None,
    }
    if not executable:
        if os.path.lexists(OSSUTIL_MANAGED_BIN):
            return {
                **base,
                "ready": False,
                "installed": True,
                "state": "unavailable",
                "code": "managed_binary_invalid",
                "installable": False,
                "path": OSSUTIL_MANAGED_BIN,
                "detail": "托管路径已存在，但不是可执行的 ossutil 文件",
            }
        if not platform_info["supported"]:
            return {
                **base,
                "ready": False,
                "code": "unsupported_platform",
                "installable": False,
                "detail": "当前平台不支持自动安装 ossutil",
            }
        return {**base, "ready": False, "detail": "未安装 ossutil，可点击状态标签自动安装"}
    try:
        result = await run([executable, "--version"], 15)
        version_line = ossutil_version_line(result)
    except HTTPException as exc:
        message = str(exc.detail).replace("\n", " ")[:240]
        return {
            **base,
            "ready": False,
            "state": "unavailable",
            "code": "version_check_failed",
            "installable": False,
            "detail": "已安装，但 ossutil --version 失败：%s" % message,
        }
    compatible_version = re.fullmatch(r"ossutil version: (v1\.\d+\.\d+)", version_line)
    if not compatible_version:
        return {
            **base,
            "ready": False,
            "state": "unavailable",
            "code": "unsupported_version",
            "installable": False,
            "version": version_line or None,
            "detail": "已安装，但版本不受支持：%s（需要 ossutil 1.x）" % (version_line or "未知版本"),
        }
    detected_version = compatible_version.group(1)
    return {
        **base,
        "ready": True,
        "usable": True,
        "state": "ready",
        "code": "ready" if detected_version == OSSUTIL_VERSION else "compatible_version",
        "installable": False,
        "version": detected_version,
        "detail": "已安装且版本可用" if detected_version == OSSUTIL_VERSION else "已安装兼容的 ossutil 1.x 版本",
    }


async def oss_readiness_data() -> Dict[str, Any]:
    ossutil_item = await ossutil_readiness_item()
    ossutil_ready = ossutil_item["ready"]
    rsync_ready = shutil.which("rsync") is not None
    config_status = inspect_oss_config()
    local_transfer_ready = ossutil_ready and config_status["valid"]
    remote_transfer_ready = config_status["valid"]
    directory_download_ready = local_transfer_ready and rsync_ready
    missing = []
    if not ossutil_ready: missing.append("ossutil（仅本机传输需要）")
    if not config_status["valid"]: missing.append("OSS 配置")
    if not rsync_ready: missing.append("rsync（本机目录下载需要）")
    return {
        "ready": local_transfer_ready,
        "missing": missing,
        "capabilities": {
            "transfer": local_transfer_ready,
            "local_transfer": local_transfer_ready,
            "remote_transfer": remote_transfer_ready,
            "sync": directory_download_ready,
            "file_transfer": local_transfer_ready,
            "directory_upload": local_transfer_ready,
            "directory_download": directory_download_ready,
        },
        "dependencies": {"ossutil": ossutil_ready, "python_tarfile": True, "rsync": rsync_ready},
        "ossutil": ossutil_item,
        "config": {**config_status, "path": "~/.ossutilconfig", "endpoint": OSS_ENDPOINT},
        "sync_root": OSS_SYNC_ROOT,
        "default_excludes": DEFAULT_OSS_EXCLUDES,
    }


async def itask_readiness_item() -> Dict[str, Any]:
    platform_info = itask_platform_info()
    target = display_home_path(ITASK_MANAGED_BIN)
    executable = resolve_itask_bin()
    base = {
        "id": "itask",
        "label": "itask",
        "feature": "容器管理",
        "installed": bool(executable),
        "usable": False,
        "state": "missing",
        "code": "not_installed",
        "installable": platform_info["supported"],
        "path": executable,
        "install_target": target,
        "platform": platform_info["platform"],
        "platform_label": platform_info["label"],
        "artifact": platform_info["artifact"],
    }
    if not executable:
        if os.path.lexists(ITASK_MANAGED_BIN):
            return {
                **base,
                "ready": False,
                "installed": True,
                "state": "unavailable",
                "code": "managed_binary_invalid",
                "installable": False,
                "path": ITASK_MANAGED_BIN,
                "detail": "托管路径已存在，但不是可执行的 itask 文件",
            }
        if not platform_info["supported"]:
            return {**base, "ready": False, "code": "unsupported_platform", "installable": False, "detail": "当前平台不支持自动安装 itask"}
        return {**base, "ready": False, "detail": "未安装 itask，可点击状态标签自动安装"}
    try:
        await run([executable, "list"], 15)
        return {**base, "ready": True, "usable": True, "state": "ready", "code": "ready", "installable": False, "detail": "已安装且服务可用"}
    except HTTPException as exc:
        message = str(exc.detail).replace("\n", " ")[:240]
        return {**base, "ready": False, "state": "unavailable", "code": "list_failed", "installable": False, "detail": "已安装，但 itask list 失败：%s" % message}


@app.get("/api/readiness")
async def readiness():
    items = [await itask_readiness_item()]

    ssh_path = shutil.which("ssh")
    items.append(dependency_item("ssh", "SSH", "终端与文件", bool(ssh_path), "已安装" if ssh_path else "未找到系统 ssh 命令"))

    sshfs_path = shutil.which("sshfs")
    items.append(dependency_item("sshfs", "sshfs / macFUSE", "本机挂载", bool(sshfs_path), "已安装" if sshfs_path else "未安装，本机挂载功能不可用"))

    items.append(await ossutil_readiness_item())
    config_status = inspect_oss_config()
    items.append(dependency_item("oss_config", "OSS 配置", "OSS", config_status["valid"], config_status["detail"]))
    rsync_path = shutil.which("rsync")
    items.append(dependency_item("rsync", "rsync", "OSS 目录同步", bool(rsync_path), "已安装" if rsync_path else "未找到系统 rsync"))

    return {
        "ready": all(item["ready"] for item in items),
        "core_ready": all(item["ready"] for item in items if item["id"] in {"itask", "ssh"}),
        "items": items,
    }


@app.post("/api/itask/install")
async def install_itask(request: Request):
    require_local_itask_install_request(request)
    async with event_loop_lock("itask_install"):
        current = await itask_readiness_item()
        if current["installed"]:
            if current["ready"]:
                return JSONResponse({"installed": False, "item": current}, headers={"Cache-Control": "no-store"})
            raise HTTPException(409, "itask 已安装但不可用，请先处理现有安装")
        if not current["installable"] or not current["artifact"]:
            raise HTTPException(409, current["detail"])

        candidate_path = ""
        published_inode = None
        try:
            candidate_path = await asyncio.to_thread(prepare_itask_candidate, current["artifact"])
            try:
                await run([candidate_path, "list"], 15)
            except HTTPException as exc:
                raise HTTPException(502, "下载的 itask 验证失败：%s" % str(exc.detail).replace("\n", " ")[:240])
            try:
                published_inode = await asyncio.to_thread(publish_itask_candidate, candidate_path)
            except OSError as exc:
                raise HTTPException(502, "无法安装 itask：%s" % exc)
            candidate_path = ""
            item = await itask_readiness_item()
            if not item["ready"]:
                await asyncio.to_thread(remove_published_itask, published_inode)
                published_inode = None
                raise HTTPException(502, item["detail"])
            return JSONResponse({"installed": True, "item": item}, headers={"Cache-Control": "no-store"})
        finally:
            if candidate_path:
                await asyncio.to_thread(cleanup_itask_candidate, candidate_path)


@app.post("/api/ossutil/install")
async def install_ossutil(request: Request):
    require_local_ossutil_install_request(request)
    async with event_loop_lock("ossutil_install"):
        lock_descriptor: Optional[int] = None
        candidate_path = ""
        published_inode: Optional[int] = None
        installation_committed = False
        try:
            lock_descriptor, cancellation_requested = await await_install_thread(acquire_ossutil_install_lock)
            if cancellation_requested:
                raise asyncio.CancelledError
            current = await ossutil_readiness_item()
            if current["installed"]:
                if current["ready"]:
                    return JSONResponse({"installed": False, "item": current}, headers={"Cache-Control": "no-store"})
                raise HTTPException(409, "ossutil 已安装但不可用，请先处理现有安装")
            if not current["installable"]:
                raise HTTPException(409, current["detail"])
            platform_info = ossutil_platform_info()
            candidate_path, cancellation_requested = await await_install_thread(prepare_ossutil_candidate, platform_info)
            if cancellation_requested:
                raise asyncio.CancelledError
            try:
                validation = await run([candidate_path, "--version"], 15)
            except HTTPException as exc:
                raise HTTPException(502, "下载的 ossutil 验证失败：%s" % str(exc.detail).replace("\n", " ")[:240])
            version_line = ossutil_version_line(validation)
            if version_line != OSSUTIL_EXPECTED_VERSION_LINE:
                raise HTTPException(
                    502,
                    "下载的 ossutil 版本不匹配：%s（需要 %s）"
                    % (version_line or "未知版本", OSSUTIL_VERSION),
                )
            try:
                published_inode, cancellation_requested = await await_install_thread(
                    publish_ossutil_candidate, candidate_path,
                )
                candidate_path = ""
                if cancellation_requested:
                    raise asyncio.CancelledError
                _, cancellation_requested = await await_install_thread(fsync_ossutil_install_dir)
                if cancellation_requested:
                    raise asyncio.CancelledError
            except OSError as exc:
                raise HTTPException(502, "无法安装 ossutil：%s" % exc)
            item = await ossutil_readiness_item()
            if not item["ready"]:
                raise HTTPException(502, item["detail"])
            installation_committed = True
            return JSONResponse({"installed": True, "item": item}, headers={"Cache-Control": "no-store"})
        finally:
            if published_inode is not None and not installation_committed:
                await await_install_thread(remove_published_ossutil, published_inode)
            if candidate_path:
                await await_install_thread(cleanup_ossutil_candidate, candidate_path)
            if lock_descriptor is not None:
                await await_install_thread(release_ossutil_install_lock, lock_descriptor)


@app.get("/api/config")
async def config():
    saved = load_gui_config()
    users = saved.get("users", [])
    images = saved.get("images", [DEFAULT_IMAGE])
    default_user = users[0] if users else ""
    default_image = images[0] if images else DEFAULT_IMAGE
    return {"defaults": {"image": default_image, "user": default_user}, "users": users, "images": images, "remote_root": REMOTE_ROOT, "local_mount_root": LOCAL_MOUNT_ROOT, "local_mount_remote_root": LOCAL_MOUNT_REMOTE_ROOT, "sshfs_available": shutil.which("sshfs") is not None}


@app.get("/api/oss/readiness")
async def oss_readiness():
    return await oss_readiness_data()


@app.get("/api/oss/config")
async def current_oss_config(request: Request):
    require_local_credentials_request(request)
    async with event_loop_lock("oss_config"):
        credentials = read_oss_credentials()
    return JSONResponse(
        content=credentials,
        headers={
            "Cache-Control": "no-store, private",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/oss/config")
async def configure_oss(request: OssConfigRequest):
    async with event_loop_lock("oss_config"):
        replaced = write_oss_config(
            request.access_key_id,
            request.access_key_secret.get_secret_value(),
            request.replace_existing,
        )
    return {"configured": True, "replaced": replaced, "readiness": await oss_readiness_data()}


@app.get("/api/oss/local-directories")
async def oss_local_directories(path: str = "", files: bool = False):
    directory = safe_local_directory(path or OSS_SYNC_ROOT, must_exist=True)
    entries = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    entries.append({"name": entry.name, "kind": "d"})
                elif files and entry.is_file():
                    entries.append({"name": entry.name, "kind": "f"})
    except OSError as exc:
        raise HTTPException(502, "无法读取本机目录：%s" % exc)
    entries.sort(key=lambda e: (e["kind"] == "f", e["name"]))
    return {"path": directory, "entries": entries}


@app.post("/api/oss/local-directories")
async def create_local_directory(request: CreateDirectoryRequest):
    parent = safe_local_directory(request.path, must_exist=True)
    target = safe_local_directory(os.path.join(parent, request.name))
    try:
        os.mkdir(target)
    except FileExistsError:
        raise HTTPException(409, "目录已存在")
    except OSError as exc:
        raise HTTPException(502, "无法创建本机目录：%s" % exc)
    return {"path": target, "created": True}


@app.get("/api/oss/objects")
async def list_oss_objects(user: str, prefix: str = ""):
    ossutil = local_ossutil()
    if not USER_PATTERN.fullmatch(user):
        raise HTTPException(400, "invalid OSS user")
    checked_prefix = safe_oss_prefix(prefix)
    result = await run(local_ossutil_args(ossutil, "ls", oss_uri(user, checked_prefix), "-d"), 60)
    entries = parse_oss_directory_listing(result["output"], user, checked_prefix)

    async def listing_for_key(key: str) -> str:
        metadata = await run(local_ossutil_args(ossutil, "ls", oss_uri(user, key)), 60)
        return metadata["output"]

    await populate_oss_object_metadata(entries, user, listing_for_key)
    return {"user": user, "prefix": checked_prefix, "entries": entries}


@app.get("/api/pods/{name}/oss/objects")
async def list_remote_oss_objects(name: str, user: str, prefix: str = ""):
    validate_name(name)
    if not USER_PATTERN.fullmatch(user):
        raise HTTPException(400, "invalid OSS user")
    checked_prefix = safe_oss_prefix(prefix)
    port = require_tunnel(name)
    tool = await remote_ossutil(name, port)
    request_id = "list_%s" % uuid.uuid4().hex
    remote_config = "/tmp/itask_gui_oss_%s.conf" % request_id
    operation_error: Optional[Exception] = None
    was_cancelled = False
    result: Optional[Dict[str, str]] = None
    entries: Optional[List[Dict[str, Any]]] = None
    try:
        remote_config = await copy_oss_config_to_remote(name, port, request_id)
        command = "%s --config-file %s ls %s -d" % (
            shlex.quote(tool), shlex.quote(remote_config), shlex.quote(oss_uri(user, checked_prefix)),
        )
        result = await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)], 60)
        entries = parse_oss_directory_listing(result["output"], user, checked_prefix)

        async def listing_for_key(key: str) -> str:
            metadata_command = "%s --config-file %s ls %s" % (
                shlex.quote(tool), shlex.quote(remote_config), shlex.quote(oss_uri(user, key)),
            )
            metadata = await run(
                ssh_args(name, port) + ["sh", "-lc", shlex.quote(metadata_command)], 60,
            )
            return metadata["output"]

        await populate_oss_object_metadata(entries, user, listing_for_key)
    except asyncio.CancelledError:
        was_cancelled = True
    except Exception as exc:
        operation_error = exc
    try:
        await await_uninterruptible(remove_remote_oss_config(name, port, remote_config))
    except Exception as exc:
        raise HTTPException(
            502,
            "Pod OSS 对象列表操作结束，但临时凭据清理失败，可能存在残留：%s"
            % str(getattr(exc, "detail", exc)),
        )
    if was_cancelled:
        raise asyncio.CancelledError
    if operation_error is not None:
        raise operation_error
    if result is None or entries is None:
        raise HTTPException(502, "Pod OSS 对象列表未返回结果")
    return {"user": user, "prefix": checked_prefix, "entries": entries}


@app.delete("/api/oss/objects")
async def delete_oss_object(user: str, key: str, recursive: bool = False):
    ossutil = local_ossutil()
    if not USER_PATTERN.fullmatch(user):
        raise HTTPException(400, "invalid OSS user")
    checked_key = safe_oss_prefix(key) if recursive else safe_oss_key(key)
    if recursive and not checked_key:
        raise HTTPException(400, "不能删除 OSS 用户根目录")
    uri = oss_uri(user, checked_key)
    args = local_ossutil_args(ossutil, "rm", uri)
    if recursive:
        args.append("-r")
    args.append("-f")
    await run(args, 3600 if recursive else 60)
    return {"deleted": True, "key": checked_key, "recursive": recursive}


async def queue_oss_transfer(request: OssUnifiedTransferRequest, pod: Optional[str] = None) -> Dict[str, Any]:
    validate_transfer_request(request)
    if pod is not None:
        validate_name(pod)
    async with event_loop_lock("oss_task_create"):
        cleanup_old_oss_states()
        if current_oss_task_id():
            raise HTTPException(409, "已有正在运行的 OSS 任务")
        if pod is not None:
            require_tunnel(pod)
        task_id = uuid.uuid4().hex
        state: Dict[str, Any] = {
            "task_id": task_id,
            "status": "queued",
            "phase": "queued",
            "message": "传输任务已排队",
            "log": [],
            "side": "remote" if pod else "local",
            "type": "transfer",
            "direction": request.direction,
            "user": request.user,
            "source_kind": request.source_kind,
            "package_upload": request.package_upload,
            "extract_download": request.extract_download,
            "statistics": {"files": None, "bytes": None, "skipped": 0},
        }
        if request.direction == "download":
            state["progress"] = {
                "bytes": 0,
                "total": None,
                "percent": None,
                "speed": 0,
                "started_at": None,
            }
        if pod is not None:
            state["pod"] = pod
        oss_states[task_id] = state
        if pod is None:
            oss_tasks[task_id] = asyncio.create_task(run_oss_local_transfer(task_id, request))
        else:
            oss_tasks[task_id] = asyncio.create_task(run_oss_remote_transfer(task_id, pod, request))
        return state


@app.post("/api/oss/transfers")
async def start_local_unified_transfer(request: OssUnifiedTransferRequest):
    return await queue_oss_transfer(request)


@app.post("/api/pods/{name}/oss/transfers")
async def start_remote_unified_transfer(name: str, request: OssUnifiedTransferRequest):
    return await queue_oss_transfer(request, name)


@app.get("/api/oss/active-task")
async def active_oss_task():
    task_id = current_oss_task_id()
    return {"task": oss_states.get(task_id) if task_id else None}


@app.get("/api/oss/tasks/{task_id}")
async def oss_task_status(task_id: str):
    state = oss_states.get(task_id)
    if not state: raise HTTPException(404, "OSS 任务不存在")
    return state


@app.delete("/api/oss/tasks/{task_id}")
async def cancel_oss_task(task_id: str):
    state = oss_states.get(task_id)
    if not state: raise HTTPException(404, "OSS 任务不存在")
    task = oss_tasks.get(task_id)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError): await task
    return oss_states[task_id]



@app.get("/api/pods")
async def pods():
    try:
        return await list_itask_pods()
    except HTTPException as exc:
        return JSONResponse({"output": "", "pods": [], "warning": exc.detail})


@app.get("/api/pods/{name}/details")
async def pod_details(name: str):
    validate_name(name)
    result = await run([require_itask_bin(), "describe", name], 60)
    return {"name": name, "details": result["output"], "warning": result["error"] or None}


@app.post("/api/pods")
async def create_pod(request: CreatePod):
    result = await run([require_itask_bin(), "create", "--name", request.name, "--nocard", "--image", request.image, "--type", "a3", "--skip-sync", "--hostnet", "--user", request.user], 300)
    saved = load_gui_config()
    users = saved.get("users", [])
    images = saved.get("images", [DEFAULT_IMAGE])
    changed = False
    if request.user and request.user not in users:
        users.append(request.user); changed = True
    if request.image and request.image not in images:
        images.append(request.image); changed = True
    if changed:
        save_gui_config({**saved, "users": users, "images": images})
    mount_states[request.name] = {"status": "waiting_for_pod", "message": "Pod 已创建，正在等待调度；就绪后自动挂载 NAS"}
    if AUTO_MOUNT_NAS: mount_tasks[request.name] = asyncio.create_task(mount_nas(request.name))
    else: mount_states[request.name] = {"status": "skipped", "message": "NAS mount skipped"}
    return {"name": request.name, "output": result["output"], "mount_status": mount_states[request.name]["status"]}


async def mount_nas(name: str):
    encoded = base64.b64encode(NAS_SCRIPT.encode()).decode()
    command = "echo %s | base64 -d > /tmp/itask_gui_nas.sh && chmod 700 /tmp/itask_gui_nas.sh && bash /tmp/itask_gui_nas.sh" % encoded
    last_error = "Pod 尚未就绪"
    try:
        for attempt in range(NAS_MOUNT_RETRIES):
            try:
                listing = await list_itask_pods()
                pod = next((item for item in listing["pods"] if item.get("name") == name), None)
                lifecycle = str((pod or {}).get("pod_status") or (pod or {}).get("status") or "unknown")
                normalized_status = lifecycle.strip().lower()
                if normalized_status in {"failed", "error", "stopped"}:
                    mount_states[name] = {
                        "status": "error",
                        "message": "Pod 状态为 %s，未执行 NAS 挂载；请查看任务详情" % lifecycle,
                    }
                    break
                if normalized_status not in {"running", "ready"}:
                    last_error = "Pod 状态为 %s" % lifecycle if pod else "任务尚未出现在列表中"
                    mount_states[name] = {
                        "status": "waiting_for_pod",
                        "message": "%s，NAS 尚未开始挂载（%d/%d）" % (last_error, attempt + 1, NAS_MOUNT_RETRIES),
                        "pod_status": lifecycle,
                    }
                    if attempt + 1 < NAS_MOUNT_RETRIES:
                        await asyncio.sleep(NAS_MOUNT_RETRY_INTERVAL)
                        continue
                    mount_states[name] = {
                        "status": "error",
                        "message": "等待 Pod 就绪超时：%s；请查看任务详情" % last_error,
                    }
                    break
                mount_states[name] = {"status": "mounting", "message": "Pod 已就绪，正在挂载 NAS"}
                result = await run([require_itask_bin(), "exec", name, "--tty=false", "--", "bash", "-lc", command], 600)
                mount_states[name] = {"status": "ready", "message": "NAS mount complete", "output": result["output"]}
                break
            except Exception as exc:
                last_error = str(getattr(exc, "detail", exc))
                mount_states[name] = {
                    "status": "mounting",
                    "message": "NAS 挂载失败，准备重试 %d/%d：%s" % (attempt + 1, NAS_MOUNT_RETRIES, last_error[:240]),
                }
                if attempt + 1 < NAS_MOUNT_RETRIES:
                    await asyncio.sleep(NAS_MOUNT_RETRY_INTERVAL)
                else:
                    mount_states[name] = {"status": "error", "message": "NAS mount failed: " + last_error}
    finally:
        mount_tasks.pop(name, None)


@app.get("/api/pods/{name}/mount-status")
async def mount_status(name: str):
    validate_name(name); return mount_states.get(name, {"status": "unknown", "message": "no mount task"})


@app.delete("/api/pods/{name}")
async def delete_pod(name: str):
    validate_name(name)
    for tid, state in list(oss_states.items()):
        if state.get("pod") == name and state.get("status") in ("queued", "running"):
            task = oss_tasks.get(tid)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            break
    for remote_path in list(local_mount_paths.get(name, {})):
        await unmount_local_record(name, remote_path)
    for registry in (local_mount_processes, local_mount_paths, local_mount_created, local_mount_ports):
        registry.pop(name, None)
    local_mount_preferred_ports.pop(name, None)
    save_local_mount_state()
    await stop_tunnel(name)
    task = mount_tasks.pop(name, None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    mount_states.pop(name, None)
    result = await run([require_itask_bin(), "delete", name, "--force"], 300)
    return {"name": name, "output": result["output"]}


@app.post("/api/pods/{name}/tunnel")
async def tunnel(name: str):
    validate_name(name)
    preserved_port = active_local_mount_port(name)
    if name in tunnels:
        process = tunnels[name]
        if process.returncode is None:
            port = tunnel_ports[name]
            if await check_tunnel_ssh(name, port):
                return {"name": name, "port": port, "running": True}
            if preserved_port is None:
                preserved_port = port if active_local_mount_port(name) else None
        await stop_tunnel(name)
    port = preserved_port or free_port()
    process = await asyncio.create_subprocess_exec(require_itask_bin(), "ssh-tunnel", name, "--port", str(port), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env())
    tunnels[name], tunnel_ports[name] = process, port
    try:
        await wait_tunnel_ready(port)
    except HTTPException:
        stderr = (await process.stderr.read()).decode(errors="replace").strip() if process.returncode is not None else ""
        await stop_tunnel(name)
        raise HTTPException(502, stderr or "隧道建立失败，请稍后重试")
    return {"name": name, "port": port, "running": True}


async def stop_tunnel(name: str):
    process = tunnels.pop(name, None); port = tunnel_ports.pop(name, None)
    if process and process.returncode is None: process.terminate(); await process.wait()
    if port is not None:
        # Shut down the SSH control master tied to this tunnel; with
        # ControlPersist=yes it would otherwise linger until the connection drops.
        control_path = ssh_control_path(name, port)
        with contextlib.suppress(Exception):
            await run(["ssh", "-O", "exit", "-S", control_path, "-p", str(port), "%s@127.0.0.1" % SSH_USER], 5)
        with contextlib.suppress(OSError):
            os.unlink(control_path)


def require_tunnel(name: str) -> int:
    validate_name(name); process = tunnels.get(name)
    if not process or process.returncode is not None: raise HTTPException(409, "tunnel is not ready")
    return tunnel_ports[name]


@app.get("/api/pods/{name}/local-mount")
async def local_mount_info(name: str):
    validate_name(name)
    return local_mount_status(name)


class LocalMountRequest(BaseModel):
    remote_path: str = "/"
    local_path: str = ""


@app.post("/api/pods/{name}/local-mount")
async def local_mount(name: str, request: LocalMountRequest):
    validate_name(name)
    if not shutil.which("sshfs"):
        raise HTTPException(501, "未检测到 sshfs。请先安装并授权 macFUSE，然后执行：brew install gromgit/fuse/sshfs-mac")
    remote_path = safe_remote_directory(request.remote_path or LOCAL_MOUNT_REMOTE_ROOT)
    requested_path = request.local_path.strip()
    local_path = normalize_local_mount_path(requested_path or local_mount_path(name, remote_path))
    async with event_loop_lock("local_mount:%s" % name):
        existing_path = local_mount_paths.get(name, {}).get(remote_path)
        if existing_path is not None and existing_path != local_path:
            if os.path.ismount(existing_path):
                raise HTTPException(409, "该 Pod 目录已挂载到 %s，请先卸载后再更换位置" % existing_path)
            await unmount_local_record(name, remote_path)
        elif existing_path == local_path and os.path.ismount(local_path):
            return {
                "remote_path": remote_path,
                "local_path": local_path,
                "mounted": True,
                "state_persisted": save_local_mount_state(),
            }
        elif existing_path == local_path:
            await unmount_local_record(name, remote_path)

        for other_name, paths in local_mount_paths.items():
            for other_remote_path, other_local_path in paths.items():
                if other_local_path == local_path:
                    raise HTTPException(
                        409,
                        "该本地目录已登记给 %s:%s，请先卸载原记录" % (other_name, other_remote_path),
                    )
        if os.path.ismount(local_path):
            raise HTTPException(409, "该本地目录已被其他系统挂载占用，未自动复用")
        if os.path.lexists(local_path) and not os.path.isdir(local_path):
            raise HTTPException(400, "本地挂载位置已存在且不是目录")
        created_by_app = not os.path.exists(local_path)
        try:
            os.makedirs(local_path, mode=0o755, exist_ok=True)
        except OSError as exc:
            raise HTTPException(502, "无法创建本地挂载目录：%s" % exc)
        port = int((await tunnel(name))["port"])
        try:
            process = await start_local_mount(name, local_path, remote_path, port)
        except HTTPException as exc:
            if created_by_app:
                with contextlib.suppress(OSError):
                    os.rmdir(local_path)
            raise HTTPException(502, "SSHFS 挂载失败：%s" % exc.detail)
        if not os.path.ismount(local_path):
            await terminate_local_mount_process(process)
            if created_by_app:
                with contextlib.suppress(OSError):
                    os.rmdir(local_path)
            raise HTTPException(502, "SSHFS 命令已结束，但未发现挂载目录")
        local_mount_processes.setdefault(name, {})[remote_path] = process
        local_mount_paths.setdefault(name, {})[remote_path] = local_path
        local_mount_created.setdefault(name, {})[remote_path] = created_by_app
        local_mount_ports.setdefault(name, {})[remote_path] = port
        local_mount_preferred_ports[name] = port
        track_local_mount_process(name, remote_path, process)
        persisted = save_local_mount_state()
        return {
            "remote_path": remote_path,
            "local_path": local_path,
            "mounted": True,
            "state_persisted": persisted,
        }


@app.delete("/api/pods/{name}/local-mount")
async def local_unmount(name: str, remote_path: str = "", force: bool = False):
    validate_name(name)
    async with event_loop_lock("local_mount:%s" % name):
        if remote_path:
            checked_remote_path = safe_remote_directory(remote_path)
            await unmount_local_record(name, checked_remote_path, force)
        else:
            for checked_remote_path in list(local_mount_paths.get(name, {})):
                await unmount_local_record(name, checked_remote_path, force)
    return local_mount_status(name)


@app.post("/api/local-mount/open")
async def open_in_finder(path: str = ""):
    if not path or not os.path.isdir(path):
        raise HTTPException(400, "无效路径")
    await asyncio.create_subprocess_exec("open", path)
    return {"ok": True}


@app.post("/api/local-mount/pick")
async def pick_local_folder():
    script_env = os.environ.copy()
    if (script_env.get("LANG") or "").startswith("C."):
        try:
            proc = await asyncio.create_subprocess_exec(
                "defaults", "read", "-g", "AppleLanguages",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), 5)
            if "zh" in stdout.decode(errors="replace").lower():
                script_env["LANG"] = "zh_CN.UTF-8"
        except Exception:
            pass
    try:
        process = await asyncio.create_subprocess_exec(
            "osascript", "-e", 'POSIX path of (choose folder with prompt "选择本地挂载目录")',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=script_env,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), 60)
        if process.returncode:
            return {"path": None}
        raw_path = stdout.decode(errors="replace").strip()
        path = "/" if raw_path == "/" else raw_path.rstrip("/")
        return {"path": path or None}
    except Exception:
        return {"path": None}


@app.get("/api/pods/{name}/files")
async def files(name: str, path: str = "", show_hidden: bool = False):
    port = require_tunnel(name); directory = remote_path(path)
    quoted = shlex.quote(directory)
    # The first find only surfaces top-level errors (missing or unreadable
    # directory); the second lists the directory plus each subdirectory's
    # children in one pass. The previous per-entry -exec check forked one
    # shell per entry, which dominated latency on large directories.
    # Traversal errors below the top level are swallowed, so unreadable
    # subdirectories simply report empty=true. Hidden filtering happens in
    # the parser so that hidden children still count towards non-empty,
    # matching the old semantics.
    batch_command = (
        "find {0} -mindepth 1 -maxdepth 1 >/dev/null || exit 1; "
        "find {0} -mindepth 1 -maxdepth 2 -printf '%P\t%y\t%s\n' 2>/dev/null || true"
    ).format(quoted)
    batch_result = await run(ssh_args(name, port) + ["sh", "-c", shlex.quote(batch_command)])
    return {"path": directory, "entries": parse_file_tree_entries(batch_result["output"], show_hidden)}


def attachment_filename(filename: str) -> str:
    return "attachment; filename*=UTF-8''%s" % quote(filename, safe="")


async def inspect_download_target(name: str, path: str, kind: str) -> Dict[str, Any]:
    port = require_tunnel(name)
    target = remote_path(path)
    quoted_target = shlex.quote(target)
    if kind == "file":
        command = (
            "if ! test -f {0}; then echo __NOT_FILE__ >&2; exit 44; fi; "
            "if ! test -r {0}; then echo __NOT_READABLE__ >&2; exit 45; fi; "
            "wc -c < {0}"
        ).format(quoted_target)
    elif kind == "directory":
        if target == "/":
            raise HTTPException(400, "不支持下载整个根文件系统，请选择具体目录")
        command = (
            "if ! test -d {0}; then echo __NOT_DIRECTORY__ >&2; exit 44; fi; "
            "if ! test -r {0} || ! test -x {0}; then echo __NOT_READABLE__ >&2; exit 45; fi; "
            "if ! command -v tar >/dev/null 2>&1; then echo __TAR_MISSING__ >&2; exit 46; fi; "
            "printf ready"
        ).format(quoted_target)
    else:
        raise HTTPException(400, "invalid download kind")
    try:
        result = await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)])
    except HTTPException as exc:
        detail = str(exc.detail)
        if "__NOT_FILE__" in detail:
            raise HTTPException(400, "只能下载普通文件")
        if "__NOT_DIRECTORY__" in detail:
            raise HTTPException(400, "目标不是目录")
        if "__NOT_READABLE__" in detail:
            raise HTTPException(403, "没有读取目标的权限")
        if "__TAR_MISSING__" in detail:
            raise HTTPException(501, "Pod 中缺少 tar，无法打包目录")
        raise
    filename = posixpath.basename(target.rstrip("/")) or "download"
    size = int(result["output"].strip()) if kind == "file" else None
    return {"port": port, "target": target, "filename": filename, "size": size, "kind": kind}


async def remote_download_response(name: str, port: int, command: str, filename: str, media_type: str):
    try:
        process = await asyncio.create_subprocess_exec(
            *ssh_args(name, port), "sh", "-lc", shlex.quote(command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env(),
        )
    except FileNotFoundError:
        raise HTTPException(501, "command not found: ssh")
    except OSError as exc:
        raise HTTPException(502, "无法启动下载：%s" % exc)

    async def read_stderr_tail(limit: int = 64 * 1024) -> bytes:
        tail = bytearray()
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                break
            tail.extend(chunk)
            if len(tail) > limit:
                del tail[:-limit]
        return bytes(tail)

    async def stream():
        stderr_task = asyncio.create_task(read_stderr_tail())
        try:
            while True:
                chunk = await process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
            await process.wait()
            stderr = await stderr_task
            if process.returncode:
                detail = stderr.decode(errors="replace").strip()
                raise RuntimeError("远端下载中断：%s" % (detail or "SSH 命令执行失败"))
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    headers = {
        "Content-Disposition": attachment_filename(filename),
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(stream(), media_type=media_type, headers=headers)


@app.get("/api/pods/{name}/files/download-info")
async def download_info(name: str, path: str, kind: str = "file"):
    info = await inspect_download_target(name, path, kind)
    filename = info["filename"] + (".tar.gz" if kind == "directory" else "")
    return {"path": info["target"], "kind": kind, "filename": filename, "size": info["size"]}


@app.get("/api/pods/{name}/files/download")
async def download(name: str, path: str):
    info = await inspect_download_target(name, path, "file")
    command = "cat -- %s" % shlex.quote(info["target"])
    return await remote_download_response(
        name, info["port"], command, info["filename"], "application/octet-stream")


@app.get("/api/pods/{name}/directories/download")
async def download_directory(name: str, path: str):
    info = await inspect_download_target(name, path, "directory")
    parent = posixpath.dirname(info["target"].rstrip("/")) or "/"
    basename = posixpath.basename(info["target"].rstrip("/"))
    command = "tar -C %s -czf - -- %s" % (shlex.quote(parent), shlex.quote(basename))
    return await remote_download_response(
        name, info["port"], command, info["filename"] + ".tar.gz", "application/gzip")


@app.post("/api/pods/{name}/directories")
async def create_remote_directory(name: str, request: CreateDirectoryRequest):
    port = require_tunnel(name)
    parent = safe_remote_directory(request.path)
    target = safe_remote_directory(posixpath.join(parent, request.name))
    command = "if test -e %s; then echo exists >&2; exit 17; fi; mkdir -- %s" % (shlex.quote(target), shlex.quote(target))
    try:
        await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)])
    except HTTPException as exc:
        if "exists" in str(exc.detail): raise HTTPException(409, "目录已存在")
        raise
    return {"path": target, "created": True}


@app.post("/api/pods/{name}/files/upload")
async def upload(name: str, path: str, file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES: raise HTTPException(413, "file too large")
    process = await asyncio.create_subprocess_exec(*ssh_args(name, require_tunnel(name)), "sh", "-c", "cat > " + shlex.quote(remote_path(path)), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, env=env())
    _, stderr = await process.communicate(data)
    if process.returncode: raise HTTPException(502, stderr.decode(errors="replace") or "upload failed")
    return {"path": path, "size": len(data)}


@app.delete("/api/pods/{name}/files")
async def delete_file(name: str, path: str):
    target = remote_path(path)
    if target in {"/", REMOTE_ROOT}:
        raise HTTPException(400, "不能删除根目录")
    process = await asyncio.create_subprocess_exec(
        *ssh_args(name, require_tunnel(name)),
        "sh", "-lc", shlex.quote("if test -d %s; then rmdir -- %s; else rm -f -- %s; fi" % (shlex.quote(target), shlex.quote(target), shlex.quote(target))),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env(),
    )
    _, stderr = await process.communicate()
    if process.returncode:
        raise HTTPException(502, stderr.decode(errors="replace") or "delete failed")
    return {"path": path, "deleted": True}


@app.post("/api/pods/{name}/files/rename")
async def rename_file(name: str, request: RenameRequest):
    port = require_tunnel(name)
    old_path = remote_path(request.path)
    if old_path in {"/", REMOTE_ROOT}:
        raise HTTPException(400, "不能重命名根目录")
    parent = posixpath.dirname(old_path)
    new_path = posixpath.join(parent, request.name)
    command = "if test -e %s; then echo exists >&2; exit 17; fi; mv -- %s %s" % (
        shlex.quote(new_path), shlex.quote(old_path), shlex.quote(new_path))
    try:
        await run(ssh_args(name, port) + ["sh", "-lc", shlex.quote(command)])
    except HTTPException as exc:
        if "exists" in str(exc.detail): raise HTTPException(409, "目标名称已存在")
        raise
    return {"old_path": old_path, "new_path": new_path, "renamed": True}


@app.websocket("/ws/pods/{name}/terminal")
async def terminal(websocket: WebSocket, name: str):
    await websocket.accept()
    process = None
    output_task = None
    try:
        process = await asyncio.create_subprocess_exec(*ssh_args(name, require_tunnel(name)), "-tt", "bash", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env())
        async def output():
            try:
                while True:
                    data = await process.stdout.read(4096)
                    if not data:
                        break
                    if websocket.client_state != WebSocketState.CONNECTED:
                        break
                    await websocket.send_text(data.decode(errors="replace"))
            except (asyncio.CancelledError, RuntimeError):
                pass
        output_task = asyncio.create_task(output())
        while True:
            process.stdin.write((await websocket.receive_text()).encode()); await process.stdin.drain()
    except Exception as e:
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await websocket.send_text(f"\r\n终端连接异常：{e}\r\n")
    finally:
        if output_task:
            output_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await output_task
        if process and process.returncode is None:
            process.terminate()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()
        if websocket.client_state == WebSocketState.CONNECTED: await websocket.close()


async def cancel_background_tasks(tasks: Dict[str, asyncio.Task]) -> None:
    running = [task for task in tasks.values() if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)


@app.on_event("shutdown")
async def shutdown():
    await cancel_background_tasks(oss_tasks)
    await cancel_background_tasks(mount_tasks)
    for name in list(local_mount_paths):
        for remote_path in list(local_mount_paths.get(name, {})):
            with contextlib.suppress(HTTPException):
                await unmount_local_record(name, remote_path)
    for name in list(tunnels):
        await stop_tunnel(name)


if __name__ == "__main__":
    import uvicorn

    args = parse_gui_args()
    if args.config is not None:
        config_file = os.path.abspath(os.path.expanduser(args.config))
        if not os.path.isfile(config_file):
            raise SystemExit("config file not found: %s" % config_file)
        GUI_CONFIG_FILE = config_file
        # Persist CLI host/port so the config file reflects how the app runs.
        if args.host is not None or args.port is not None:
            saved = load_gui_config()
            if args.host is not None:
                saved["host"] = args.host
            if args.port is not None:
                saved["port"] = args.port
            save_gui_config(saved)
        GUI_SETTINGS = load_gui_settings()
        AUTO_MOUNT_NAS = GUI_SETTINGS["auto_mount_nas"]
    if args.auto_mount_nas is not None:
        AUTO_MOUNT_NAS = args.auto_mount_nas
    host = args.host if args.host is not None else GUI_SETTINGS["host"]
    port = args.port if args.port is not None else GUI_SETTINGS["port"]
    uvicorn.run(app, host=host, port=port)
