#!/usr/bin/env bash
# 状态备份:SQLite 一致性快照 + 可读快照打包。
# 建议加 crontab(工作日收盘后):
#   0 16 * * 1-5  /path/to/workspace/scripts/backup.sh >> /path/to/workspace/data/logs/backup.log 2>&1
# 备份文件建议异地化:scp 到另一台机器 / 传 OSS / 推私有 git 仓库均可。
set -euo pipefail
cd "$(dirname "$0")/.."
ts=$(date +%Y%m%d_%H%M%S)
mkdir -p data/backups

if [ -f data/qtrader.db ]; then
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 data/qtrader.db ".backup 'data/backups/qtrader_${ts}.db'"
  else
    cp data/qtrader.db "data/backups/qtrader_${ts}.db"
  fi
fi

tar czf "data/backups/state_${ts}.tar.gz" \
  --exclude='data/cache' --exclude='data/backups' --exclude='data/logs' \
  data 2>/dev/null || true

# 只保留最近 30 份
ls -1t data/backups/state_*.tar.gz 2>/dev/null | tail -n +31 | xargs -r rm -f
ls -1t data/backups/qtrader_*.db 2>/dev/null | tail -n +31 | xargs -r rm -f

echo "[$(date '+%F %T')] 备份完成: data/backups/state_${ts}.tar.gz"
