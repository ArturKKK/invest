# 🏗️ PROJECT NOTES — AI Crypto Trading System

**Последнее обновление:** 2026-03-08  
**Автор / контекст для AI-ассистента:** Этот файл — справочник для восстановления контекста.

---

## 1. Инфраструктура

### VPS (боевой сервер)
- **IP:** `185.42.163.63`
- **OS:** Ubuntu 22.04, 2 cores, 4GB RAM, 25GB SSD (~8GB свободно)
- **SSH:** `ssh root@185.42.163.63`
- **User:** `trader` (рабочий юзер для бота)
- **Рабочая директория:** `/home/trader/invest/`
- **Python:** 3.10.12, venv в `/home/trader/invest/venv/`
- **VPN:** уже работает на сервере
- **Swap:** 1GB (создан вручную)
- **Timezone:** UTC

### Локальная машина
- **macOS M3 Pro**
- **Python:** 3.13
- **Проект:** `/Users/a.s.tabakov/Developer/invest/`
- **venv:** `.venv/` (локальный)

### systemd сервис
- **Файл:** `/etc/systemd/system/crypto-trader.service`
- **Запуск:** `systemctl start|stop|restart crypto-trader`
- **Логи stdout:** `/home/trader/invest/logs/bot.log`
- **Логи stderr:** `/home/trader/invest/logs/bot_error.log`
- **journalctl:** `journalctl -u crypto-trader -f` (только systemd events, python output — в файлах)
- **Параметры:** `--mode paper --loop --capital 5000 --rebal 24`

### Деплой (как обновить бота)
```bash
# С локалки:
scp run_trading.py root@185.42.163.63:/home/trader/invest/
ssh root@185.42.163.63 "chown trader:trader /home/trader/invest/run_trading.py && systemctl restart crypto-trader"
# Проверить:
ssh root@185.42.163.63 "sleep 60 && tail -60 /home/trader/invest/logs/bot.log"
```

---

## 2. API-ключи и аккаунты

### OKX Demo Trading
- **Сайт:** https://www.okx.com/demo-trading
- **Режим аккаунта:** Single-currency margin (acctLv=2) — ПЕРЕКЛЮЧЕНО ВРУЧНУЮ на сайте!
- **Position mode:** net_mode (one-way)
- **API Key:** `932bb5fb-c534-4c4c-95e1-6e24e6215440`
- **Secret:** `3E47539C27A58F64A288C3ED6CCB396E`
- **Passphrase:** `Starz7z7z7!`
- **Начальный баланс:** $5,000 USDT (demo)
- **Плечо:** 3x cross margin
- **⚠️ ВАЖНО:** Режим аккаунта нельзя переключить через API на демо — только через веб-интерфейс!

### Telegram Bot
- **Bot Token:** `8699291703:AAFK5Azv4AtJAUxXiRT3BwYSnDp_VVPZFbM`
- **Chat ID:** `548740796`
- **Бот:** @<ваш_бот> (создан через @BotFather)
- **Команды:** `/status`, `/pnl`, `/help`

### CryptoCompare (новости)
- **API Key:** `4e5a45e5329dd0ad1ead03d9be06b286f07acd4a442f60ec7992e9cef3e8b4bd`

### Все ключи также в `.env` (и на VPS `/home/trader/invest/.env`)

---

## 3. Торговая система (Champion Config)

### Модели (15 моделей, ансамбль)
| Набор | Папка | Моделей | Описание |
|-------|-------|---------|----------|
| LGB v6 | `results_v6/` | 5 seeds | 12h target, edge-boost sizing |
| LGB v7 | `results_v7/` | 5 seeds | Blended target (4h+12h+24h) |
| CatBoost | `results_catboost/` | 5 seeds | CatBoost, те же фичи |

### Бэктест результаты (champion: ensemble 3x 24h)
- **Sharpe:** 8.04
- **Return:** +37.2%
- **Profit Factor:** 2.85
- **Win Rate:** 67%
- **Max DD:** -18.7%

### Risk Config (DEFAULT_RISK в run_trading.py)
```python
DEFAULT_RISK = {
    'n_long': 5,           # 5 лонгов
    'n_short': 5,          # 5 шортов
    'kelly_frac': 1.0,     # 100% Kelly
    'vol_target': 0.008,   # 0.8% vol target
    'vol_lookback': 30,
    'dd_stop': -0.20,      # Стоп при -20% DD
    'dd_resume': -0.08,    # Возобновить при -8% DD
    'leverage': 3,         # 3x плечо
    'confidence_threshold': 0.0,
}
```

### Как работает бот (run_trading.py --mode paper --loop)
1. Каждые 24ч (в 00:05 UTC) запускается цикл
2. Скачивает 800ч OHLCV данных для 50 монет с Binance
3. Строит 147 фичей (TA, cross-asset, sentiment proxies)  
4. Ранкирует фичи cross-sectionally
5. Генерирует сигналы через 15 моделей (средний score)
6. Строит портфель: top-5 long + bottom-5 short, по $1500 каждая (с 3x)
7. Diff-rebalancing: сравнивает с текущими позициями, меняет только нужное
8. Логирует в `trading_logs/trade_YYYYMMDD_HHMM.json`

---

## 4. Заблокированные монеты на OKX Demo

**Только для торговли!** Для обучения моделей используются все 50 монет.

В `_OKX_BLOCKED` (run_trading.py) — монеты, которые нельзя торговать на OKX demo:

| Монета | Причина |
|--------|---------|
| MATIC, UNI, APT, FTM, MANA | Не существуют на OKX demo swaps |
| RUNE, EGLD, FLOW, SNX, ENJ | Не существуют на OKX demo swaps |
| BAT, ONE, ICX, ENS, GALA, GRT | Не существуют на OKX demo swaps |
| CHZ, MKR | 51155 compliance restricted |
| ZIL | 51202 max market order exceeded |

**19 из 50** заблокированы → торгуем на **31 монете**.
Фильтрация происходит в `construct_portfolio()` — blocked монеты не попадают в top/bottom 5.

---

## 5. Telegram — что приходит

### Автоматические сообщения
| Событие | Когда | Что |
|---------|-------|-----|
| 🚀 Startup | При запуске бота | Mode, capital, risk config |
| 📊 Positions | Каждый ребаланс | Таблица позиций (symbol, side, USD, score) |
| ✅/❌ Fills | После ордеров | Какие ордера прошли/упали |
| 💰 Cycle PnL | После settlement (sim) | P&L цикла, equity, DD |
| ⚠️ DD Warning | DD < -15% | Предупреждение о просадке |
| 🔴 DD Stop | DD < -20% | Торговля остановлена |
| 🟢 DD Resume | DD восстановился > -8% | Торговля возобновлена |
| 📈 Daily Summary | Раз в сутки | Equity, total PnL, win rate, позиции |
| ❌ Error | При ошибке | Stack trace / описание |
| 🛑 Shutdown | При остановке | Причина остановки |

### Команды (ты пишешь боту)
| Команда | Ответ |
|---------|-------|
| `/status` | Текущие позиции, equity, DD, uptime |
| `/pnl` | Подробная статистика: total PnL, win rate, avg win/loss |
| `/help` | Список команд |

---

## 6. Файловая структура (важные файлы)

### Скрипты
- `run_trading.py` — **главный бот** (paper/live trading)
- `telegram_bot.py` — модуль Telegram (алерты + команды)
- `run_pipeline_v6.py` — обучение LGB v6
- `run_pipeline_v7.py` — обучение LGB v7
- `run_pipeline_catboost.py` — обучение CatBoost
- `run_ensemble.py` / `run_ensemble_v2.py` — оценка ансамблей
- `run_fast_sim.py` — быстрая симуляция (бэктест с risk overlay)
- `fetch_crypto_news.py` — скачивание новостей (CryptoCompare + GDELT)
- `_fill_gaps.py` — заполнение пропусков в новостных данных

### Конфиги и документация
- `.env` — все секреты (НЕ в git)
- `.env.example` — шаблон .env (в git)
- `PROGRESS.md` — история исследований (от v1 до champion)
- `PROJECT_NOTES.md` — **этот файл** (инфра, ключи, памятка)
- `RESULTS.md` — результаты бэктестов
- `RFC_TRADING_SYSTEM.md` — исходный RFC архитектуры

### Деплой
- `deploy/crypto-trader.service` — systemd unit
- `deploy/deploy.sh` — скрипт деплоя
- `deploy/setup_vps.sh` — первоначальная настройка VPS

### Модели (всё в git, синхронизировано на VPS)
- `results_v6/` — LGB v6 модели (5 × .txt)
- `results_v7/` — LGB v7 модели (5 × .txt)
- `results_catboost/` — CatBoost модели (5 × .cb)
- `results_v5/` — LGB v5 (старый, не используется в trading)

### Логи (на VPS)
- `/home/trader/invest/logs/bot.log` — stdout бота
- `/home/trader/invest/logs/bot_error.log` — stderr
- `/home/trader/invest/trading_logs/` — JSON логи каждого цикла
- `/home/trader/invest/trading_logs/trades.csv` — **CSV-лог всех ставок** (per-trade)

---

## 7. Известные ограничения

1. **Missing features:** LGB моделям не хватает 5 фичей, CatBoost — 9 (sentiment/news features, заполняются нулями)
2. **News pipeline не запущен на VPS** — скрипт `fetch_crypto_news.py` + cron для GDELT/CryptoCompare
3. **OKX Demo баланс:** потеряли ~$500 на ранних тестах с неправильным sizing → текущий баланс ~$4,485
4. **Маржа:** С 3x плечом на $4,485 — максимум ~$13,455. На 10 позиций по $1,500 нужно $15,000 → последняя позиция может не влезть
5. **OKX Demo ≠ Production:** на реальном OKX другие инструменты, больше доступных монет

---

## 8. Что делать если контекст пропал

### Быстрый старт для AI-ассистента:
1. Прочитай `PROJECT_NOTES.md` (этот файл) — вся инфра и конфиг
2. Прочитай `PROGRESS.md` — история моделей и исследований
3. `run_trading.py` — основной бот (~1550 строк)
4. `telegram_bot.py` — Telegram модуль (~340 строк)
5. Проверь статус: `ssh root@185.42.163.63 "systemctl status crypto-trader --no-pager"`
6. Логи: `ssh root@185.42.163.63 "tail -50 /home/trader/invest/logs/bot.log"`
7. Стейт: `ssh root@185.42.163.63 "cat /home/trader/invest/trading_logs/trading_state.json"`

### Деплой после изменений:
```bash
# Проверить синтаксис
python3 -c "import py_compile; py_compile.compile('run_trading.py', doraise=True)"
# Залить + рестарт
scp run_trading.py root@185.42.163.63:/home/trader/invest/
ssh root@185.42.163.63 "chown trader:trader /home/trader/invest/run_trading.py && systemctl restart crypto-trader"
```

---

## 9. Хронология изменений

### 2026-03-08 — Production Launch
- Интегрирован Telegram bot в `run_trading.py`
- Исправлен OKX order sizing (USD → contracts через ctVal)
- Добавлен retry с экспоненциальным backoff
- Diff-based rebalancing (не пересоздаёт позиции которые не изменились)
- `generate_signal()` исправлен: загружает 15 моделей (v6+v7+CatBoost) вместо только v5
- `DEFAULT_RISK` обновлён до champion config
- Leverage 3x добавлен в `construct_portfolio()`
- OKX: переключено на ccxt unified symbols (`BTC/USDT:USDT`), `tdMode: cross`, `mgnMode: cross`
- 19 монет заблокированы (не существуют на demo / compliance / max order)
- Фильтр blocked монет в `construct_portfolio()` чтобы не тратить слоты
- VPS настроен, systemd сервис, деплой-скрипты
- Добавлен per-trade CSV лог с отслеживанием win/loss
- Первый успешный ребаланс: 9/10 позиций открыты ✅
