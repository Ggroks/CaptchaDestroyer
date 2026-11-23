# HTTP API сканера цифр

Отправьте изображение в поле `file`. Сервис передаст его существующему распознавателю и вернёт четыре цифры в поле `text`. Значение `"0042"` можно сразу использовать как строку.

[← К проекту](README.md) · **[Локальный запуск](#локальный-запуск)** · [Примеры клиентов](#примеры-клиентов) · [Коды ответов](#коды-ответов) · [VPS и Docker](#развёртывание-на-vps)

| Метод и путь | Что возвращает | Авторизация |
| --- | --- | --- |
| `POST /api/recognize` | Результат OCR для одного изображения | Заголовок `X-API-Key` |
| `GET /health` | `{"status":"ok"}`, если API отвечает и файл модели на месте | Свободный доступ |
| `GET /docs` | Swagger UI для отправки запросов из браузера | Ключ вводится для вызова OCR |

## Локальный запуск

Откройте терминал в корне репозитория. Понадобятся Python 3.14 и модель по пути `tessdata/eng.traineddata`. Если окружение для CLI уже создано, переходите к установке `requirements-api.txt`. [Скачать модель](https://github.com/tesseract-ocr/tessdata_best/blob/4.1.0/eng.traineddata)

**1. Подготовьте окружение.** При первой настройке скопируйте `.env.example` в `.env`. Последняя команда в каждом примере генерирует ключ.

<details open>
<summary><strong>Linux · Bash</strong></summary>

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-api.txt
cp -n .env.example .env
.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))"
```

</details>

<details>
<summary><strong>Windows · PowerShell</strong></summary>

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

</details>

**2. Задайте ключ.** Впишите сгенерированное значение в строку `OCR_API_KEY=` файла `.env`. Этот же ключ понадобится приложению, которое отправляет изображения.

**3. Запустите Uvicorn.** Это HTTP-сервер, который принимает запросы для FastAPI. Оставьте его работающим и откройте второй терминал для запросов.

```bash
# Linux
.venv/bin/python -m uvicorn ocr_api:create_app --factory --env-file .env
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m uvicorn ocr_api:create_app --factory --env-file .env
```

Базовый адрес — `http://127.0.0.1:8000`. Откройте `/docs` по этому адресу, нажмите **Authorize**, введите ключ и выберите **Try it out** у `POST /api/recognize`. Если ключ и настройки уже заданы в окружении процесса, параметр `--env-file .env` можно опустить.

## Первый запрос

В Bash задайте переменную `OCR_API_KEY` в терминале клиента, затем отправьте приложенный к проекту PNG. В PowerShell используйте `$env:OCR_API_KEY`.

```bash
curl -sS -H "X-API-Key: $OCR_API_KEY" \
  -F "file=@docs/images/digits-0042.png" \
  http://127.0.0.1:8000/api/recognize
```

```powershell
curl.exe -sS -H "X-API-Key: $env:OCR_API_KEY" -F "file=@docs/images/digits-0042.png" http://127.0.0.1:8000/api/recognize
```

**Пример ответа · HTTP 200**

```json
{
  "success": true,
  "text": "0042",
  "processing_ms": 794
}
```

`processing_ms` измеряет время после приёма файла, включая запуск процесса OCR. Если распознаватель вернул четыре цифры с низкой уверенностью, API также передаст их в `text`. Отдельного поля уверенности в ответе нет.

| Параметр загрузки | Допустимое значение |
| --- | --- |
| Тело запроса | `multipart/form-data` с одним файлом в поле `file` |
| Формат | Однокадровый PNG, JPEG, WebP или BMP |
| Размер файла | До 5 МиБ по умолчанию |
| Разрешение | До 16 млн пикселей |

## Примеры клиентов

Примеры читают ключ из переменной окружения клиента `OCR_API_KEY` и используют изображение из репозитория. Запускайте их из корня проекта. Для VPS замените базовый адрес на HTTPS-адрес сервера.

<details>
<summary><strong>Python · httpx</strong></summary>

Установите клиент командой `python -m pip install httpx`.

```python
import os
import httpx

with open("docs/images/digits-0042.png", "rb") as image:
    response = httpx.post(
        "http://127.0.0.1:8000/api/recognize",
        headers={"X-API-Key": os.environ["OCR_API_KEY"]},
        files={"file": ("digits.png", image, "image/png")},
        timeout=20,
    )

result = response.json()
if response.is_success:
    print(result["text"])
else:
    print(response.status_code, result["error"])
```

</details>

<details>
<summary><strong>JavaScript · Node.js 22+</strong></summary>

Сохраните пример как `recognize.mjs` и запустите `node recognize.mjs`.

```javascript
import { readFile } from "node:fs/promises";

const image = await readFile("docs/images/digits-0042.png");
const form = new FormData();
form.set("file", new Blob([image], { type: "image/png" }), "digits.png");

const response = await fetch("http://127.0.0.1:8000/api/recognize", {
  method: "POST",
  headers: { "X-API-Key": process.env.OCR_API_KEY },
  body: form,
  signal: AbortSignal.timeout(20_000),
});

const result = await response.json();
if (response.ok) {
  console.log(result.text);
} else {
  console.error(response.status, result.error);
  process.exitCode = 1;
}
```

</details>

<details>
<summary><strong>PHP · cURL</strong></summary>

Включите расширение cURL. Этот вызов можно использовать из серверного PHP-скрипта.

```php
<?php
$request = curl_init('http://127.0.0.1:8000/api/recognize');
curl_setopt_array($request, [
    CURLOPT_HTTPHEADER => ['X-API-Key: ' . getenv('OCR_API_KEY')],
    CURLOPT_POSTFIELDS => ['file' => new CURLFile('docs/images/digits-0042.png', 'image/png')],
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT => 20,
]);
$body = curl_exec($request);
if ($body === false) {
    throw new RuntimeException(curl_error($request));
}
$status = curl_getinfo($request, CURLINFO_RESPONSE_CODE);
$result = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
echo $status === 200 ? $result['text'] : $result['error'];
```

</details>

При вызове из веб-интерфейса отправляйте запрос со своего сервера, где хранится ключ.

## Коды ответов

Если четыре цифры прочитать не удалось, сервис вернёт HTTP 422 с таким телом.

```json
{
  "success": false,
  "text": null,
  "error": "Recognition failed"
}
```

| HTTP-код | Причина |
| --- | --- |
| `200` | Распознавание успешно. Для `/health` ответ содержит `{"status":"ok"}`. |
| `400` | Некорректная форма или отсутствует файл в поле `file`. |
| `401` | Ключ отсутствует или неверен. |
| `413` | Превышен лимит размера файла либо разрешения. |
| `415` | Неподдерживаемый тип запроса или формат изображения. |
| `422` | Изображение повреждено, пусто или распознаватель не смог вернуть четыре цифры. |
| `500` | Внутренняя ошибка OCR. Подробности доступны в журнале сервера. |
| `503` | Все процессы OCR заняты. Заголовок `Retry-After: 1` указывает время до повтора в секундах. Для `/health` код означает, что файл модели отсутствует. |
| `504` | Истёк тайм-аут. Сервер завершил процесс для этого изображения. |

## Настройки

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `OCR_API_KEY` | Обязательна | Секрет для `X-API-Key`. Печатные символы ASCII без пробелов. |
| `OCR_MAX_UPLOAD_BYTES` | `5242880` | Максимальный размер файла в байтах. Для всего тела запроса дополнительно выделено 64 КиБ на заголовки и границы multipart. |
| `OCR_TIMEOUT_SECONDS` | `10` | Максимальное время работы процесса OCR в секундах. |
| `OCR_MAX_CONCURRENCY` | `2` | Максимум одновременных процессов OCR на один процесс Uvicorn. |
| `OCR_TESSDATA_DIR` | `tessdata/` рядом с кодом | Каталог с `eng.traineddata`. |

API применяет настройки при запуске. После изменения ключа или лимитов перезапустите процесс Uvicorn.

## Развёртывание на VPS

Перенесите проект на Linux VPS с Docker. Перед сборкой скопируйте текущий `eng.traineddata` в `tessdata/`. Создайте `.env` по образцу `.env.example` и запишите в него ключ.

```bash
docker build -t digit-ocr-api .
docker run -d --name digit-ocr-api --init --restart unless-stopped \
  --env-file .env -p 127.0.0.1:8000:8000 digit-ocr-api
curl -fsS http://127.0.0.1:8000/health
```

Docker будет перезапускать сервис при сбое и после перезагрузки VPS. Журнал доступен через `docker logs digit-ocr-api`.

Для доступа с других серверов направьте DNS-имя на VPS и настройте HTTPS-прокси. Например, в Caddyfile укажите свой домен и оставьте порт 8000 доступным только локально. Для автоматического HTTPS Caddy должны быть доступны входящие порты 80 и 443. [Документация Caddy](https://caddyserver.com/docs/quick-starts/https)

```caddy
ocr.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Проверьте `https://ocr.example.com/health`. Затем замените локальный адрес в примерах на `https://ocr.example.com/api/recognize`.

## Тесты

Из корня проекта установите тестовый HTTP-клиент и запустите набор тестов.

```bash
# Linux
.venv/bin/python -m pip install httpx
.venv/bin/python -m unittest discover -s tests -v
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Тесты генерируют изображения в памяти и проверяют распознавание с ведущими нулями, параллельные запросы, ключи, форматы файлов, лимиты и тайм-аут. Для проверки сбоя OCR тест подставляет повреждённую модель из временного каталога.
