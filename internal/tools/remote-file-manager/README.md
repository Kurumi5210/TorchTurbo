# 远程文件管理

一个运行在本机的 Web 工具，用于管理远端文件，通过 itask 容器和 OSS 对象存储方式实现在浏览器中对远端文件浏览、上传和下载；同时提供 Pod 申请/连接、网页终端、SSHFS 本机挂载等配套功能。

## 可选：SSHFS（仅“本机挂载”功能需要）

只有需要把 Pod 目录挂载到本机时才需要安装，其余功能不受影响（目前仅支持 macOS）：

```sh
brew install --cask macfuse
brew install gromgit/fuse/sshfs-mac
```

按 macOS 提示允许 macFUSE 系统扩展并按需重启。

## 快速开始

要求：Python 3.9+。

```sh
git clone -b v3 https://code.alipay.com/Language_Compiler/torchturbo.git
cd torchturbo/internal/tools/remote-file-manager
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/config_example.json config.json
python3 app.py -h 127.0.0.1 -p 8765 -f config.json
```

启动后访问 <http://127.0.0.1:8765>。停止服务按 `Ctrl+C`。

| 参数 | 说明 |
| --- | --- |
| `-f <路径>` | 指定配置文件路径，指定后同次启动的 `-h`/`-p` 写回 `host`、`port`；创建 Pod 时新使用的用户名、镜像自动追加写入 `users`、`images` |
| `-h` | Web 服务监听地址 |
| `-p` | Web 服务端口 |


## 配置参考

配置文件 config.json 由快速开始中的 `cp` 命令从 `config/config_example.json` 创建，可按需修改其中的值；配置仅在启动时通过 `-f` 指定后生效

```jsonc
{
  // Web 服务监听地址（缺省为 127.0.0.1）
  "host": "127.0.0.1",
  // Web 服务监听端口（缺省为 8765）
  "port": 8765,
  // 用户名
  "users": [
    "your_employee_id"
  ],
  // 镜像列表
  "images": [
    "hcr.meta-wulan01.hw-wulan.local/antsys/vllm:release_0.18.0_0415_202604171255_aarch64"
  ],
  // 新 Pod 是否自动安装 NFS 工具并挂载 NAS（缺省为 true）
  "auto_mount_nas": true
}
```
